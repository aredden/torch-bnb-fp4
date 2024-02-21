from enum import Enum
from functools import partial
from math import prod
from typing import List, Union

import torch
import torch.nn.functional as F
from bitsandbytes import functional as BF
from bitsandbytes.nn.modules import Linear4bit, Params4bit
from torch import nn
from torch_bnb_fp4_ext import ScalarType as ScalarType_  # type: ignore
from torch_bnb_fp4_ext import dequantize_fp4 as dequantize_fp4_  # type: ignore
from torch_bnb_fp4_ext import gemv_4bit_inference_impl  # type: ignore
from torch_bnb_fp4_ext import qlinear as qlinear_  # type: ignore
from torch_bnb_fp4_ext import qlinear_bias as qlinear_bias_  # type: ignore

try:
    from fused_dense_cuda import linear_bias_forward  # type: ignore
except ImportError:
    print(
        f"Couldn't import fused_dense_cuda, if you want to use it, please install the nvidia 'apex' package."
    )
    linear_bias_forward = torch.nn.functional.linear


class ScalarType(Enum):
    bfloat16 = ScalarType_.bfloat16
    float16 = ScalarType_.float16
    float32 = ScalarType_.float32

    @classmethod
    def from_torch_dtype(cls, dtype: torch.dtype):
        if dtype == torch.bfloat16:
            return cls.bfloat16
        elif dtype == torch.float16:
            return cls.float16
        elif dtype == torch.float32:
            return cls.float32
        else:
            raise ValueError(f"Unsupported dtype {dtype}")

    @classmethod
    def from_str(cls, dtype: str):
        if dtype == "bfloat16":
            return cls.bfloat16
        elif dtype == "float16":
            return cls.float16
        elif dtype == "float32":
            return cls.float32
        else:
            raise ValueError(f"Unsupported dtype {dtype}")

    @property
    def torch_dtype(self):
        if self == ScalarType.BFloat16:
            return torch.bfloat16
        elif self == ScalarType.Float16:
            return torch.float16
        elif self == ScalarType.Float32:
            return torch.float32
        else:
            raise ValueError(f"Unsupported dtype {self}")


@torch.no_grad
def dequantize_fp4(
    qweight: torch.ByteTensor,
    absmax: torch.Tensor,
    blocksize: int,
    M: int,
    N: int,
    dtype=torch.float16,
):
    return dequantize_fp4_(
        qweight, absmax, blocksize, M, N, ScalarType.from_torch_dtype(dtype).value
    )


@torch.no_grad
def gemm_4bit_inference(
    A: torch.Tensor,
    B: torch.Tensor,
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    dtype=torch.float16,
    Bshape=None,
):
    return gemv_4bit_inference_impl(
        A, B, absmax, code, blocksize, ScalarType.from_torch_dtype(dtype).value, Bshape
    )


@torch.no_grad
def gemm_4bit_inference_qtype(
    A: torch.Tensor,
    B: torch.Tensor,
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    dtype: ScalarType = ScalarType.bfloat16.value,
    Bshape: List[int] = None,
):
    return gemv_4bit_inference_impl(A, B, absmax, code, blocksize, dtype, Bshape)


@torch.no_grad
def dequantize_fp4_qtype(
    qweight: torch.ByteTensor,
    absmax: torch.Tensor,
    blocksize: int,
    M: int,
    N: int,
    dtype: ScalarType = ScalarType.bfloat16.value,
):
    return dequantize_fp4_(
        qweight,
        absmax,
        blocksize,
        M,
        N,
        dtype,
    )


class QuantData:
    def __init__(self, A, state: BF.QuantState, shape, bias=None, original_lin=None):
        self.A = A
        self.absmax = state.absmax.float()
        self.blocksize = state.blocksize
        self.M = shape[0]
        self.N = shape[1]
        self.code = state.code.float()
        self.o_type = None
        self.qtype = None
        self.quant_state = state
        self.bias = original_lin.bias if hasattr(original_lin, "bias") else bias
        self.original_lin = original_lin
        self.compute_dtype_set = False
        self.forward_func = linear_bias_forward if self.bias is not None else F.linear

    def set_compute_type(self, x):
        self.o_type = x.dtype
        self.qtype = ScalarType.from_torch_dtype(x.dtype).value
        if self.bias is not None:
            self.bias = self.bias.to(dtype=self.o_type)
        self.compute_dtype_set = True

    def dequantize(self):
        return dequantize_fp4_(
            self.A,
            self.absmax,
            self.blocksize,
            self.M,
            self.N,
            self.qtype,
        )

    def qgemv(self, A: torch.Tensor):
        return gemm_4bit_inference_qtype(
            A=A,
            B=self.A.t(),
            absmax=self.absmax,
            code=self.code,
            blocksize=self.blocksize,
            dtype=self.qtype,
            Bshape=self.quant_state.shape,
        )

    def qlinear(self, A: torch.Tensor):
        if self.bias is None:
            return qlinear_(
                A,
                self.A,
                self.absmax,
                self.M,
                self.N,
                self.blocksize,
            )
        else:
            return qlinear_bias_(
                A,
                self.A,
                self.absmax,
                self.M,
                self.N,
                self.blocksize,
                self.bias,
            )

    def gemm(self, A: torch.Tensor):
        prodshape = prod(A.shape)
        is_contig = A.is_contiguous()
        if prodshape == 0:
            B_shape = self.quant_state.shape
            if A.shape[-1] == B_shape[0]:
                return torch.empty(
                    A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device
                )
            else:
                return torch.empty(
                    A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device
                )
        if not self.compute_dtype_set:
            self.set_compute_type(A)
        if prodshape == A.shape[-1]:
            if A.shape[-1] % self.blocksize != 0:
                out = self.qlinear(A)
            else:
                if not is_contig:
                    A = A.contiguous()
                # gemm 4bit only works when the input is a single batch,
                # with 1 token and batch size 1
                # aka- (1, 1, hidden_dim)
                # or (1, hidden_dim)

                if A.ndim == 3:
                    N_batch = A.shape[0]
                    A = A.view(-1, A.shape[-1])
                    out = self.qgemv(A)
                    out = out.view(N_batch, 1, -1)
                    if self.bias is not None:
                        out += self.bias
                elif A.ndim == 2:
                    out = self.qgemv(A)
                    if self.bias is not None:
                        out += self.bias
                else:
                    out = self.qlinear(A)
        else:
            out = self.qlinear(A)
        return out


class LinearHijack(nn.Module):
    def __init__(self, lin: Union[nn.Linear, Linear4bit]):
        super().__init__()
        self.lin = [lin]
        if isinstance(lin.weight, Params4bit):
            if lin.weight.quant_state is None:
                self.construct_qweights()
            else:
                self.quant_data = QuantData(
                    lin.weight.data,
                    lin.weight.quant_state,
                    lin.weight.quant_state.shape,
                    bias=lin.bias,
                    original_lin=lin,
                )
        elif not hasattr(lin.weight, "quant_state") or lin.weight.quant_state is None:
            self.construct_qweights()

    def construct_qweights(self):
        q, state = BF.quantize_fp4(self.lin[0].weight.data)
        self.quant_data = QuantData(
            q, state, self.lin[0].weight.shape, self.lin[0].bias, self.lin[0]
        )

    def forward(self, x):
        return self.quant_data.gemm(x)

    def __repr__(self):
        return f"ZippyFP4Linear(in_features={self.lin[0].in_features}, out_features={self.lin[0].out_features}, bias={self.lin[0].bias is not None})"
