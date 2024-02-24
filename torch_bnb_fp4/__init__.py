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
    """
    Enum encapsulating c++ bound torch scalar types for fp32, fp16, and bf16.
    """

    bfloat16 = ScalarType_.bfloat16
    float16 = ScalarType_.float16
    float32 = ScalarType_.float32

    @classmethod
    def from_torch_dtype(
        cls, dtype: torch.dtype
    ) -> Union["ScalarType.bfloat16", "ScalarType.float16", "ScalarType.float32"]:
        """
        Convert a torch dtype to a ScalarType.

        Args:
            dtype (torch.dtype): The torch dtype to be converted.

        Returns:
            Union[ScalarType.bfloat16, ScalarType.float16, ScalarType.float32]: The corresponding ScalarType.
        """
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
        """
        Convert a string to a ScalarType.

        Args:
            dtype (str): The string to be converted.

        Returns:
            Union[ScalarType.bfloat16, ScalarType.float16, ScalarType.float32]: The corresponding ScalarType.
        """
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
    """
    Dequantizes 4-bit quantized weights to floating-point representation.

    This function is designed to convert the 4-bit quantized weights back into their original
    floating-point format. Allows for reduced model size and potentially faster computation on
    compatible hardware, while still being able to perform operations in the model's original
    precision.

    Parameters:
    - qweight (torch.ByteTensor): The quantized weights, stored in a byte tensor.
    - absmax (torch.Tensor): The maximum absolute value of the weights, used for scaling during dequantization.
    - blocksize (int): The size of the block used for quantization. This affects how the weights were originally quantized.
    - M (int): The first dimension of the weight matrix.
    - N (int): The second dimension of the weight matrix.
    - dtype (torch.dtype, optional): The target data type for the dequantized weights. Defaults to torch.float16.

    Returns:
    - torch.FloatTensor: The dequantized weights, converted back to floating-point representation.

    The function internally calls a CUDA implementation `dequantize_fp4_` with the appropriate scalar type
    derived from the given dtype to perform the dequantization. This operation is performed without
    gradient tracking to ensure it is purely computational and does not affect backpropagation.
    """
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
    """
    Dequantizes 4-bit quantized weights to floating-point representation using codebook.

    This function is designed to convert the 4-bit quantized weights back into their original
    floating-point format. Allows for reduced model size and potentially faster computation on
    compatible hardware, while still being able to perform operations in the model's original
    precision.

    Parameters:
    - qweight (torch.ByteTensor): The quantized weights, stored in a byte tensor.
    - absmax (torch.Tensor): The maximum absolute value of the weights, used for scaling during dequantization.
    - code (torch.FloatTensor): The 16 element codebook used for dequantization.
    - blocksize (int): The size of the block used for quantization. This affects how the weights were originally quantized.
    - M (int): The first dimension of the weight matrix.
    - N (int): The second dimension of the weight matrix.
    - numel (int): The number of elements in the weight matrix.
    - qtype (torch_bnb_fp4.ScalarType): The quantization type.

    Returns:
    - torch.FloatTensor: The dequantized weights, converted back to floating-point representation using codebook.

    The function internally calls a CUDA implementation `dequantize_fp4_codebook_` with the appropriate scalar type
    derived from the given qtype to perform the dequantization.
    """
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
    """
    Dequantizes 4-bit quantized weights to floating-point representation using codebook and invokes the CUDA implementation.

    This function is designed to convert the 4-bit quantized weights back into their original
    floating-point format. Allows for reduced model size and potentially faster computation on
    compatible hardware, while still being able to perform operations in the model's original
    precision.

    Parameters:
    - qweight (torch.ByteTensor): The quantized uint8 fp4 weights, stored as a byte tensor,
        each byte represents two four bit weight indices in the codebook (code).
    - absmax (torch.Tensor): The maximum absolute value of the weights, 1 absmax per blocksize weights,
        used for scaling during dequantization.
    - code (torch.FloatTensor): The 16 element codebook used for dequantization.
    - blocksize (int): The number of elements per absmax.
    - M (int): The first dimension of the weight matrix.
    - N (int): The second dimension of the weight matrix.
    - numel (int): The number of elements in the weight matrix.
    - qtype (ScalarType): The quantization type.

    Returns:
    - torch.FloatTensor: The dequantized weights, converted back to floating-point representation using codebook.

    The function internally calls a CUDA implementation `dequantize_fp4_codebook_` with the appropriate scalar type
    derived from the given qtype to perform the dequantization.
    """
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
    B: torch.ByteTensor,
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    dtype=torch.float16,
    Bshape=None,
) -> torch.FloatTensor:
    """
    Performs 4-bit quantized matrix multiplication using a GEMV algorithm.

    This function is designed to perform matrix multiplication on 4-bit quantized matrices using the GEMM algorithm.
    It takes two input matrices A and B, the maximum absolute value of the weights (absmax), the codebook used for
    dequantization (code), the size of the block used for quantization (blocksize), the data type for the output (dtype),
    and the shape of matrix B (Bshape).

    Parameters:
    - A (torch.Tensor): The first input matrix, of shape (1, hidden) or (1, 1, hidden), where the last dimension is always
        equal to the total number of it's elements.
    - B (torch.ByteTensor): The quantized uint8 fp4 weights, stored as a byte tensor, each byte represents two four bit weight
        indices in the codebook.
    - absmax (torch.Tensor): The maximum absolute value of the weights, used for scaling during dequantization.
    - code (torch.Tensor): The 16 element codebook used for dequantization.
    - blocksize (int): The size of the block used for quantization. This affects how the weights were originally quantized.
    - dtype (torch.dtype): The data type for the output.
    - Bshape (List[int]): The shape of matrix B.

    Returns:
    - torch.FloatTensor: The result of the matrix multiplication, in the specified data type.

    The function internally calls a CUDA implementation `gemv_fp4_` with the appropriate scalar type
    derived from the given dtype to perform the matrix multiplication.
    """
    return gemv_fp4_(
        A, B, absmax, code, blocksize, ScalarType.from_torch_dtype(dtype).value, Bshape
    )


@torch.no_grad
def gemm_4bit_inference_qtype(
    A: torch.Tensor,
    B: torch.ByteTensor,
    absmax: torch.FloatTensor,
    code: torch.FloatTensor,
    blocksize: int,
    dtype: ScalarType = ScalarType.bfloat16.value,
    Bshape: List[int] = None,
) -> torch.FloatTensor:
    """
    Performs 4-bit quantized matrix multiplication using a GEMV algorithm.

    This function is designed to perform matrix multiplication on 4-bit quantized matrices using the GEMM algorithm.
    It takes two input matrices A and B, the maximum absolute value of the weights (absmax), the codebook used for
    dequantization (code), the size of the block used for quantization (blocksize), the data type for the output (dtype),
    and the shape of matrix B (Bshape).

    Parameters:
    - A (torch.Tensor): The first input matrix, of shape (1, hidden) or (1, 1, hidden), where the last dimension is always
        equal to the total number of it's elements.
    - B (torch.ByteTensor): The quantized uint8 fp4 weights, stored as a byte tensor, each byte represents two four bit weight
        indices in the codebook.
    - absmax (torch.Tensor): The maximum absolute value of the weights, used for scaling during dequantization.
    - code (torch.Tensor): The 16 element codebook used for dequantization.
    - blocksize (int): The size of the block used for quantization. This affects how the weights were originally quantized.
    - dtype (torch.dtype): The data type for the output.
    - Bshape (List[int]): The original shape of the unquantized matrix B.

    Returns:
    - torch.FloatTensor: The result of the matrix multiplication, in the specified data type.

    The function internally calls a CUDA implementation `gemv_fp4_` with the appropriate scalar type
    derived from the given dtype to perform the matrix multiplication.
    """
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
    """
    Dequantizes the 4-bit quantized weights.

    This function is designed to dequantize the 4-bit quantized weights.
    It takes the quantized weights (qweight), the maximum absolute value of the weights (absmax),
    the size of the block used for quantization (blocksize), the number of rows in the matrix (M),
    the number of columns in the matrix (N), and the data type for the output (dtype).

    Parameters:
    - qweight (torch.ByteTensor): The quantized uint8 fp4 weights, stored as a byte tensor, each byte represents two four bit weight
        indices in the codebook.
    - absmax (torch.Tensor): The maximum absolute value of the weights, used for scaling during dequantization.
    - blocksize (int): The size of the block used for quantization. This affects how the weights were originally quantized.
    - M (int): The number of rows in the matrix.
    - N (int): The number of columns in the matrix.
    - dtype (torch.dtype): The data type for the output.

    Returns:
    - torch.FloatTensor: The dequantized weights, in the specified data type.

    The function internally calls a CUDA implementation `dequantize_fp4_` with the appropriate scalar type
    derived from the given dtype to perform the dequantization.
    """
    return dequantize_fp4_(
        qweight,
        absmax,
        blocksize,
        M,
        N,
        dtype,
    )


class QuantData:
    """
    This class is used to store quantized data and implements the forward pass of a quantized linear layer.
    """

    def __init__(
        self,
        A: torch.ByteTensor,
        state: BF.QuantState,
        shape: Tuple[int, int],
        original_lin: nn.Linear,
        bias: Optional[torch.FloatTensor] = None,
        use_codebook_dequant: Optional[bool] = True,
        allow_reduced_precision_linear: Optional[bool] = False,
    ):
        """
        Initializes the QuantData class.

        This function is used to initialize the QuantData class.
        It takes the quantized data (A), the quantization state (bitsandbytes.functional.QuantState), the shape of the data (shape),
        the bias (bias), the original linear layer (original_lin), a flag to use codebook dequantization (use_codebook_dequant),
        a flag to allow reduced precision linear (allow_reduced_precision_linear), and the type of reduced precision linear dequantization (reduced_precision_linear_dequant_type).

        Parameters:
        - A `(torch.ByteTensor)` `REQUIRED` : The quantized data, stored as a byte tensor.
        - state `(bitsandbytes.functional.QuantState)` `REQUIRED` : The quantization state.
        - shape `(Tuple[int, int])` `REQUIRED` : The shape of the data.
        - original_lin `(nn.Linear)` `REQUIRED` : The original linear layer.
        - bias `(Optional[torch.FloatTensor])` `default: None` : The bias of the original linear layer, not necessary if original_lin has bias.
        - use_codebook_dequant `(Optional[bool])` `default: True` : A flag to use codebook dequantization vs fp4 tree dequantization, which is the bitsandbytes default.
        - allow_reduced_precision_linear `(Optional[bool])` `default: False` : A flag to allow reduced precision linear, will speed up full gemm (not gemv) forwards at the expense of loss of precision.
            * Typically ~0.35 elementwise error for matmul vs between ~0.04 to ~0.06 elementwise error when `False`. I do not recommend using this in general.
            * It is only applicable for input shapes where `(B, L, H), L > 1 or B > 1` or `(B, H), B > 1`, other types of gemms will remain with low elementwise error.

        Returns:
        - None
        """
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
            if self.use_codebook_dequant:
                self.qlinear = self._qlinear_low_precision_codebook
            else:
                self.qlinear = self._qlinear_low_precision_normal
        else:
            self.qlinear = self._dequant_linear
        if self.use_codebook_dequant:
            self.dequantize = self._dequantize_codebook
        else:
            self.dequantize = self._dequantize_normal

    def set_compute_type(self, x: torch.Tensor) -> None:
        """
        Sets the compute type for the input tensor.

        This function is used to set the compute type for the input tensor.
        It takes the input tensor (x) and sets the output type (o_type) and quantization type (qtype) based on the input tensor's dtype.
        If the bias is not None, it also sets the bias to the output type.

        Parameters:
        - x `(torch.Tensor)` `REQUIRED` : The input tensor.

        Returns:
        - None
        """
        self.o_type = x.dtype
        self.qtype = ScalarType.from_torch_dtype(x.dtype).value
        if self.bias is not None:
            self.bias = self.bias.to(dtype=self.o_type)
        self.compute_dtype_set = True

    def _dequant_linear(self, A: torch.Tensor) -> torch.FloatTensor:
        """
        Dequantizes the input tensor and performs a linear transformation.

        This function is used to dequantize the input tensor (A) and perform a matrix multiply + add bias.
        It takes the input tensor (A) and dequantizes it using the dequantize function.
        It then performs an nn.Linear matmul+bias using the dequantized tensor and the original linear layer's bias.

        Parameters:
        - A `(torch.Tensor)` `REQUIRED` : The input tensor.

        - torch.Tensor : The output tensor after the linear transformation.
        """
        return torch.nn.functional.linear(A, self.dequantize(), self.bias)

    def _dequantize_codebook(self) -> torch.FloatTensor:
        """
        Dequantizes this QuantData's weights using the codebook.

        Used as a wrapped simplification of the dequantize_fp4_codebook_invoke_qtype function, pre-configured with the correct arguments.
        """

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
        """
        Dequantizes this QuantData's weights using the normal method.

        Used as a wrapped simplification of the dequantize_fp4_qtype function, pre-configured with the correct arguments.
        """
        return dequantize_fp4_qtype(
            self.A,
            self.absmax,
            self.blocksize,
            self.M,
            self.N,
            self.qtype,
        )

    def _qgemv(self, A: torch.Tensor) -> torch.FloatTensor:
        """
        This function performs a Quantized GEMV operation.

        It takes the input tensor (A) and performs a matrix multiply with the transposed weight tensor (self.A).
        It then quantizes the result using the absmax, code, and blocksize parameters.

        Parameters:
        - A `(torch.Tensor)` `REQUIRED` : The input tensor.

        Returns:
        - torch.Tensor : The output tensor after the quantized GEMV operation.
        """
        return gemm_4bit_inference_qtype(
            A=A,
            B=self.A.t(),
            absmax=self.absmax,
            code=self.code,
            blocksize=self.blocksize,
            dtype=self.qtype,
            Bshape=self.quant_state.shape,
        )

    def _qlinear_low_precision_normal(self, A: torch.Tensor) -> torch.FloatTensor:
        """
        Quantized nn.Linear operation using the low precision fp4 tree dequant method.
        This method is faster than the QuantData._dequant_linear method, but has a higher elementwise error.
        It is only applicable for input shapes where `(B, L, H), L > 1 or B > 1` or `(B, H), B > 1`, other types of gemms will remain with low elementwise error.

        Parameters:
        - A `(torch.Tensor)` `REQUIRED` : The input tensor.

        Returns:
        - torch.Tensor : The output tensor after the quantized GEMV operation.
        """
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

    def _qlinear_low_precision_codebook(self, A: torch.Tensor) -> torch.FloatTensor:
        """
        Quantized nn.Linear operation using the low precision codebook dequant method.
        This method is faster than the QuantData._dequant_linear method, but has a higher elementwise error.
        It is only applicable for input shapes where `(B, L, H), L > 1 or B > 1` or `(B, H), B > 1`, other types of gemms will remain with low elementwise error.

        Parameters:
        - A `(torch.Tensor)` `REQUIRED` : The input tensor.

        Returns:
        - torch.Tensor : The output tensor after the quantized GEMV operation.
        """
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
        """
        Faux nn.Linear forward pass.
        This function is used to perform a forward pass of a quantized linear layer.
        It takes the input tensor (A) and performs a quantized matmul+bias operation using the quantized weights and bias.
        If the input tensor is not contiguous, it will be made contiguous before the operation.

        Special Cases Handled:
        - If the input tensor is not contiguous, it will be made contiguous before the operation.
        - If the input tensor shape contains a zero, the output will be a tensor of zeros with the correct (0 element) shape.
        - If the input tensor's number of elements is equal to the last dimension of itself, and is divisible by the quantized weight's block size

        Parameters:
        - A `(torch.Tensor)` `REQUIRED` : The input tensor.

        Returns:
        - torch.Tensor : The output tensor after the quantized matmul+bias operation.
        """
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
        """
        Initializes the LinearHijack class.
        This class is used to hijack a torch.nn.Linear or Linear4bit layer and replace it with a torch-bnb-fp4 version.
        It takes the original linear layer (lin) and a flag for whether to use codebook dequantization (use_codebook_dequant).
        If the original linear layer's weights are not quantized, it will quantize them using bitsandbytes quantization functions.
        If the original linear layer's weights are quantized, it will use the existing quantization state.
        """
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
        """
        Constructs the quantized weights and quantization state for the linear layer using bitsandbytes quantization functions.
        """
        q, state = BF.quantize_fp4(self.lin[0].weight.data)
        self.quant_data = QuantData(
            q,
            state,
            self.lin[0].weight.shape,
            self.lin[0],
            self.lin[0].bias,
            use_codebook_dequant=self.use_codebook_dequant,
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Calls this LinearHijack's quantized linear layer's forward method.
        If the input tensor is not contiguous, it will be made contiguous before the operation.
        If the input tensor shape contains a zero, the output will be a tensor of zeros with the correct (0 element) shape.
        If the input tensor's number of elements is equal to the last dimension of itself, and is divisible by the quantized weight's block size, an optimized GEMV operation will be used.
        If the input tensor's number of elements is not equal to the last dimension of itself, a full dequantize + matmul + bias operation will be used.

        Parameters:
        - x `(torch.Tensor)` `REQUIRED` : The input tensor.

        Returns:
        - torch.Tensor : The output tensor after the quantized matmul+bias operation.
        """
        return self.quant_data.forward(x)

    def __repr__(self) -> str:
        if hasattr(self, "quant_data"):
            return f"TorchFP4Linear(in_features={self.lin[0].in_features}, out_features={self.lin[0].out_features}, bias={self.lin[0].bias is not None}, dtype={self.quant_data.o_type})"
        else:
            return f"TorchFP4Linear(in_features={self.lin[0].in_features}, out_features={self.lin[0].out_features}, bias={self.lin[0].bias is not None})"
