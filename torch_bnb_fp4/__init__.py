from enum import Enum
from math import prod
from typing import List, Literal, Optional, Tuple, Union

import torch
from bitsandbytes import functional as BF
from bitsandbytes.nn.modules import Linear4bit, Params4bit
from torch import nn
from torch_bnb_fp4_ext import ScalarType as ScalarType_  # type: ignore
from torch_bnb_fp4_ext import dequantize_fp4 as dequantize_fp4_  # type: ignore
from torch_bnb_fp4_ext import gemv_fp4 as gemv_fp4_  # type: ignore
from torch_bnb_fp4_ext import qlinear as qlinear_  # type: ignore
from torch_bnb_fp4_ext import qlinear_bias as qlinear_bias_  # type: ignore
from torch_bnb_fp4_ext import dequantize_fp4_codebook as dequantize_fp4_codebook_  # type: ignore
from torch_bnb_fp4_ext import qlinear_codebook as qlinear_codebook_  # type: ignore
from torch_bnb_fp4_ext import qlinear_codebook_bias as qlinear_codebook_bias_  # type: ignore


class ScalarType(Enum):
    bfloat16 = ScalarType_.bfloat16
    float16 = ScalarType_.float16
    float32 = ScalarType_.float32

    @classmethod
    def from_torch_dtype(
        cls, dtype: torch.dtype
    ) -> Union["ScalarType.bfloat16", "ScalarType.float16", "ScalarType.float32"]:
        if dtype == torch.bfloat16:
            return cls.bfloat16
        elif dtype == torch.float16:
            return cls.float16
        elif dtype == torch.float32:
            return cls.float32
        else:
            raise ValueError(f"Unsupported dtype {dtype}")

    @classmethod
    def from_str(
        cls, dtype: str
    ) -> Union["ScalarType.bfloat16", "ScalarType.float16", "ScalarType.float32"]:
        if dtype == "bfloat16":
            return cls.bfloat16
        elif dtype == "float16":
            return cls.float16
        elif dtype == "float32":
            return cls.float32
        else:
            raise ValueError(f"Unsupported dtype {dtype}")

    @property
    def torch_dtype(self) -> torch.dtype:
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
) -> torch.FloatTensor:
    return dequantize_fp4_(
        qweight, absmax, blocksize, M, N, ScalarType.from_torch_dtype(dtype).value
    )


@torch.no_grad
def dequantize_fp4_codebook_invoke_qtype(
    qweight: torch.ByteTensor,
    absmax: torch.FloatTensor,
    code: torch.FloatTensor,
    blocksize: int,
    M: int,
    N: int,
    numel: int,
    qtype: ScalarType,
) -> torch.FloatTensor:
    return dequantize_fp4_codebook_(
        qweight,
        absmax,
        code,
        M,
        N,
        blocksize,
        numel,
        qtype,
    )


@torch.no_grad
def dequantize_fp4_codebook_invoke(
    qweight: torch.ByteTensor,
    absmax: torch.FloatTensor,
    code: torch.FloatTensor,
    blocksize: int,
    M: int,
    N: int,
    numel: int,
    qtype: torch.dtype,
) -> torch.FloatTensor:
    return dequantize_fp4_codebook_(
        qweight,
        absmax,
        code,
        M,
        N,
        blocksize,
        numel,
        ScalarType.from_torch_dtype(qtype).value,
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
) -> torch.FloatTensor:
    return gemv_fp4_(
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
) -> torch.FloatTensor:
    return gemv_fp4_(A, B, absmax, code, blocksize, dtype, Bshape)


@torch.no_grad
def dequantize_fp4_qtype(
    qweight: torch.ByteTensor,
    absmax: torch.Tensor,
    blocksize: int,
    M: int,
    N: int,
    dtype: ScalarType = ScalarType.bfloat16.value,
) -> torch.FloatTensor:
    return dequantize_fp4_(
        qweight,
        absmax,
        blocksize,
        M,
        N,
        dtype,
    )


class QuantData:
    def __init__(
        self,
        A: torch.ByteTensor,
        state: BF.QuantState,
        shape: Tuple[int, int],
        bias: Optional[torch.FloatTensor] = None,
        original_lin: Optional[nn.Linear] = None,
        use_codebook_dequant: Optional[bool] = False,
        allow_reduced_precision_linear: Optional[bool] = False,
        reduced_precision_linear_dequant_type: Optional[
            Literal["codebook", "bitsandbytes"]
        ] = "codebook",
    ):
        assert reduced_precision_linear_dequant_type in [
            "codebook",
            "bitsandbytes",
        ], "reduced_precision_linear_dequant_type must be either 'codebook' or 'bitsandbytes'"
        self.use_codebook_dequant = use_codebook_dequant
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
        self.numel = prod(shape)
        if allow_reduced_precision_linear:
            if reduced_precision_linear_dequant_type == "codebook":
                self.qlinear = self._qlinear_codebook
            else:
                self.qlinear = self._qlinear_normal
        else:
            self.qlinear = self._dequant_linear
        if self.use_codebook_dequant:
            self.dequantize = self._dequantize_codebook
        else:
            self.dequantize = self._dequantize_normal

    def set_compute_type(self, x: torch.Tensor) -> None:
        self.o_type = x.dtype
        self.qtype = ScalarType.from_torch_dtype(x.dtype).value
        if self.bias is not None:
            self.bias = self.bias.to(dtype=self.o_type)
        self.compute_dtype_set = True

    def _dequant_linear(self, A: torch.Tensor) -> torch.FloatTensor:
        return torch.nn.functional.linear(A, self.dequantize(), self.bias)

    def _dequantize_codebook(self) -> torch.FloatTensor:
        return dequantize_fp4_codebook_invoke_qtype(
            self.A,
            self.absmax,
            self.code,
            self.blocksize,
            self.M,
            self.N,
            self.numel,
            self.qtype,
        )

    def _dequantize_normal(self) -> torch.FloatTensor:
        return dequantize_fp4_qtype(
            self.A,
            self.absmax,
            self.blocksize,
            self.M,
            self.N,
            self.qtype,
        )

    def _qgemv(self, A: torch.Tensor) -> torch.FloatTensor:
        return gemm_4bit_inference_qtype(
            A=A,
            B=self.A.t(),
            absmax=self.absmax,
            code=self.code,
            blocksize=self.blocksize,
            dtype=self.qtype,
            Bshape=self.quant_state.shape,
        )

    def _qlinear_normal(self, A: torch.Tensor) -> torch.FloatTensor:
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

    def _qlinear_codebook(self, A: torch.Tensor) -> torch.FloatTensor:
        if self.bias is None:
            return qlinear_codebook_(
                A,
                self.A,
                self.absmax,
                self.code,
                self.M,
                self.N,
                self.blocksize,
            )
        else:
            return qlinear_codebook_bias_(
                A,
                self.A,
                self.absmax,
                self.code,
                self.M,
                self.N,
                self.blocksize,
                self.bias,
            )

    def forward(self, A: torch.FloatTensor) -> torch.FloatTensor:
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
                    out = self._qgemv(A)
                    out = out.view(N_batch, 1, -1)
                    if self.bias is not None:
                        out += self.bias
                elif A.ndim == 2:
                    out = self._qgemv(A)
                    if self.bias is not None:
                        out += self.bias
                else:
                    out = self.qlinear(A)
        else:
            out = self.qlinear(A)
        return out


class LinearHijack(nn.Module):
    def __init__(
        self, lin: Union[nn.Linear, Linear4bit], use_codebook_dequant: bool = False
    ):
        super().__init__()
        self.lin = [lin]
        self.use_codebook_dequant = use_codebook_dequant
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
                    use_codebook_dequant=self.use_codebook_dequant,
                )
        elif not hasattr(lin.weight, "quant_state") or lin.weight.quant_state is None:
            self.construct_qweights()

    def construct_qweights(self) -> None:
        q, state = BF.quantize_fp4(self.lin[0].weight.data)
        self.quant_data = QuantData(
            q,
            state,
            self.lin[0].weight.shape,
            self.lin[0].bias,
            self.lin[0],
            use_codebook_dequant=self.use_codebook_dequant,
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.quant_data.forward(x)

    def __repr__(self) -> str:
        return f"TorchFP4Linear(in_features={self.lin[0].in_features}, out_features={self.lin[0].out_features}, bias={self.lin[0].bias is not None}, dtype={self.quant_data.o_type})"
