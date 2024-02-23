import torch
from accelerate.utils.bnb import BnbQuantizationConfig, replace_with_bnb_layers
from torch import nn
from torch.utils.benchmark import Timer

from torch_bnb_fp4.__init__ import LinearHijack


def replace_with_bnb(model):

    qconfig = BnbQuantizationConfig(
        load_in_4bit=True,
        torch_dtype=torch.float16,
        bnb_4bit_compute_dtype="fp16",
        bnb_4bit_quant_type="fp4",
    )

    replace_with_bnb_layers(model, qconfig)

    return model


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


def recursively_replace(module, as_dtype=torch.float16):
    """Function to replace all bnb linear layers with torch-fp4-bnb linear layers."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(
                module,
                name,
                LinearHijack(child).to(device=child.weight.device, dtype=as_dtype),
            )
        elif isinstance(child, torch.nn.Module):
            recursively_replace(child, as_dtype=as_dtype)
    if isinstance(module, torch.nn.Linear):
        setattr(
            module,
            name,
            LinearHijack(module).to(device=module.weight.device, dtype=as_dtype),
        )


def time_run(model, inputs, label):
    timer = Timer("model(inputs)", globals=locals(), label=label)
    measure = timer.adaptive_autorange()
    return measure


def check_speed():
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
    torch.cuda.manual_seed_all(10)
    torch.manual_seed(10)
    generator = torch.Generator("cuda").manual_seed(10)
    model = TestModel(768, 2048, 4, 64).cuda().half()
    input_gemm = torch.randn(1, 768, generator=generator, device="cuda").half()
    with torch.inference_mode():
        print(time_run(model, input_gemm, "NORMAL"))
        print(time_run(model, input_gemm, "NORMAL"))
        print(time_run(model, input_gemm, "NORMAL"))

        replace_with_bnb(model)
        model.cuda()

        print(time_run(model, input_gemm, "BNB"))
        print(time_run(model, input_gemm, "BNB"))
        print(time_run(model, input_gemm, "BNB"))

        recursively_replace(model)

        print(time_run(model, input_gemm, "ZIPPY"))
        print(time_run(model, input_gemm, "ZIPPY"))
        print(time_run(model, input_gemm, "ZIPPY"))


def check():
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
    torch.cuda.manual_seed_all(10)
    torch.manual_seed(10)
    generator = torch.Generator("cuda").manual_seed(10)
    model = nn.Linear(64, 64).cuda().half()
    modelhijack = nn.Linear(64, 64).cuda().half()
    modelhijack.weight.data = model.weight.data.clone()
    modelhijack.bias.data = model.bias.data.clone()
    hijack = LinearHijack(modelhijack)
    input_gemv_3dim = torch.randn(1, 1, 64, generator=generator, device="cuda").half()
    input_gemv = torch.randn(1, 64, generator=generator, device="cuda").half()
    input_gemm_3dim = torch.randn(1, 2, 64, generator=generator, device="cuda").half()

    output_gemv_3dim = model(input_gemv_3dim)
    output_gemv_3dim_hijack = hijack(input_gemv_3dim)
    print(f"output_gemv_3dim: {output_gemv_3dim}")
    print(f"output_gemv_3dim_h: {output_gemv_3dim_hijack}")
    output_gemv = model(input_gemv)
    output_gemv_hijack = hijack(input_gemv)
    print(f"output_gemv: {output_gemv}")
    print(f"output_gemv_h: {output_gemv_hijack}")
    output_gemm_3dim = model(input_gemm_3dim)
    output_gemm_3dim_hijack = hijack(input_gemm_3dim)
    print(f"output_gemm_3dim: {output_gemm_3dim}", output_gemm_3dim.shape)
    print(
        f"output_gemm_3dim_h: {output_gemm_3dim_hijack}", output_gemm_3dim_hijack.shape
    )


if __name__ == "__main__":
    check_speed()
