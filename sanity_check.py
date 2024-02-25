import numpy as np
import torch
from accelerate.utils.bnb import BnbQuantizationConfig, replace_with_bnb_layers
from torch import nn
from torch.utils.benchmark import Timer
from prettytable import PrettyTable
from torch_bnb_fp4 import recursively_replace_with_fp4_linear


def replace_with_bnb(model, dtype=torch.float16):
    str_dtype = {
        torch.float16: "fp16",
        torch.float32: "fp32",
        torch.bfloat16: "bf16",
    }[dtype]

    qconfig = BnbQuantizationConfig(
        load_in_4bit=True,
        torch_dtype=dtype,
        bnb_4bit_compute_dtype=str_dtype,
        bnb_4bit_quant_type="fp4",
    )

    replace_with_bnb_layers(model, qconfig)

    return model


class TinyModel(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.in_proj(x)


class TestModel(nn.Module):
    def __init__(self, in_dim, hidden, num_hidden, out_dim) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(
            *([nn.GELU(), nn.Linear(hidden, hidden)] * num_hidden)
        )
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.out_proj(x)


def time_run(model, inputs, label):
    timer = Timer("model(inputs)", globals=locals(), label=label)
    measure = timer.adaptive_autorange()
    return measure


def get_avg(measurements, attribute, mul=1000000, round=5):
    return (
        np.mean([getattr(m, attribute) * mul for m in measurements]).round(round).item()
    )


def check_speed(dtype=torch.float16, gemm_type="gemm"):
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=180)
    torch.cuda.manual_seed_all(10)
    torch.manual_seed(10)
    generator = torch.Generator("cuda").manual_seed(10)
    model = TestModel(768, 2048, 4, 64).cuda().type(dtype)
    if gemm_type == "gemv":
        input_gemm = torch.randn(1, 768, generator=generator, device="cuda").type(dtype)
    else:
        input_gemm = torch.randn(2, 768, generator=generator, device="cuda").type(dtype)
    table = PrettyTable(
        field_names=["type", "mean (us)", "median (us)", "iqr (us)"],
        title=f"GEMM Speed Benchmark for {dtype} and matmul type [{gemm_type.upper()}] W/ 6 Layer MLP",
    )
    with torch.inference_mode():
        _ = time_run(model, input_gemm, "NORMAL")
        result1 = time_run(model, input_gemm, "NORMAL")
        result2 = time_run(model, input_gemm, "NORMAL")
        result_original = result1.merge([result2])
        replace_with_bnb(model, dtype=dtype)
        model.cuda()

        _ = time_run(model, input_gemm, "BNB")
        result1 = time_run(model, input_gemm, "BNB")
        result2 = time_run(model, input_gemm, "BNB")
        result_bnb = result1.merge([result2])

        model = recursively_replace_with_fp4_linear(
            model, as_dtype=dtype, device=model.in_proj.weight.device
        )

        _ = time_run(model, input_gemm, "ZIPPY")
        result1 = time_run(model, input_gemm, "ZIPPY")
        result2 = time_run(model, input_gemm, "ZIPPY")
        result_zippy = result1.merge([result2])

        result_dicts = [
            [
                "pytorch",
                get_avg(result_original, "mean"),
                get_avg(result_original, "median"),
                get_avg(result_original, "iqr"),
            ],
            [
                "bitsandbytes",
                get_avg(result_bnb, "mean"),
                get_avg(result_bnb, "median"),
                get_avg(result_bnb, "iqr"),
            ],
            [
                "torch-bnb-fp4",
                get_avg(result_zippy, "mean"),
                get_avg(result_zippy, "median"),
                get_avg(result_zippy, "iqr"),
            ],
        ]
        table.add_rows(result_dicts)
        print(table.get_string())


def simple_fwd(model, input):
    weight = model.in_proj.quant_data.dequantize()
    return torch.nn.functional.linear(input, weight, model.in_proj.quant_data.bias)


def check(dtype=torch.float16):
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=180)
    torch.cuda.manual_seed_all(10)
    torch.manual_seed(10)
    generator = torch.Generator("cuda").manual_seed(10)
    model = TinyModel(256, 256).cuda().type(dtype)
    modelhijack = TinyModel(256, 256).cuda().type(dtype)
    modelhijack.in_proj.weight.data = model.in_proj.weight.data.clone()
    modelhijack.in_proj.bias.data = model.in_proj.bias.data.clone()

    hijack = recursively_replace_with_fp4_linear(modelhijack).to("cuda", dtype=dtype)
    input_gemv_3dim = torch.randn(1, 1, 256, generator=generator, device="cuda").type(
        dtype
    )
    input_gemv = torch.randn(1, 256, generator=generator, device="cuda").type(dtype)
    input_gemm_3dim = torch.randn(
        1, 2048, 256, generator=generator, device="cuda"
    ).type(dtype)
    with torch.inference_mode():

        output_gemv_3dim = model(input_gemv_3dim)
        output_gemv_3dim_hijack = hijack(input_gemv_3dim)
        difference_avg = (output_gemv_3dim - output_gemv_3dim_hijack).abs().mean()
        print(
            "Elementwise Diff. Avg Between nn.Linear & Quant GEMV 3dim:",
            difference_avg.item(),
        )
        output_gemv = model(input_gemv)
        output_gemv_hijack = hijack(input_gemv)
        difference_avg = (output_gemv - output_gemv_hijack).abs().mean()
        print(
            "Elementwise Diff. Avg Between nn.Linear & Quant GEMV 2dim:",
            difference_avg.item(),
        )

        output_gemm_3dim = model(input_gemm_3dim)
        output_gemm_3dim_hijack = hijack(input_gemm_3dim)
        difference_avg = (output_gemm_3dim - output_gemm_3dim_hijack).abs().mean()
        print(
            "Elementwise Diff. Avg Between nn.Linear & Quant GEMM 3dim:",
            difference_avg.item(),
        )


if __name__ == "__main__":
    print("\n============ Running Sanity Checks ============\n")
    print()
    print(
        " NOTE: The acceptable range for the elementwise difference avg\n is around 0.045-0.065, which is the same as bitsandbytes.\n"
    )
    print("== Running sanity check for torch-bnb-fp4 fp32 ==\n")
    dt = torch.float32
    check_speed(dt, gemm_type="gemv")
    check_speed(dt, gemm_type="gemm")
    check(dt)
    print("\n== Running sanity check for torch-bnb-fp4 fp16 ==\n")
    dt = torch.float16
    check_speed(dt, gemm_type="gemv")
    check_speed(dt, gemm_type="gemm")
    check(dt)
    print("\n== Running sanity check for torch-bnb-fp4 bf16 ==\n")
    dt = torch.bfloat16
    check_speed(dt, gemm_type="gemv")
    check_speed(dt, gemm_type="gemm")
    check(dt)
    print("\n============= Sanity Checks Compete =============\n")
