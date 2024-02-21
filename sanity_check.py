from torch_bnb_fp4.__init__ import LinearHijack
import torch
from torch import nn


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
    check()
