#include <torch/csrc/Exceptions.h>
#include <torch/extension.h>
#include <cuda_runtime_api.h>

void dequantize_blockwise_fp4(torch::Tensor A, torch::Tensor absmax, int M, int N, int blocksize, int n, torch::Tensor out);
torch::Tensor dequantize_blockwise_codebook_fp4(
    torch::Tensor A, torch::Tensor absmax, torch::Tensor codebook, int M, int N, int blocksize, int n, torch::ScalarType dtype
);
torch::Tensor gemv_4bit_inference(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor absmax,
    torch::Tensor datatype,
    int blocksize,
    torch::ScalarType dtype,
    std::vector<uint32_t> Bshape
);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), " must be contiguous")

enum class ScalarTypeEnum {
    float16,
    float32,
    bfloat16,
};

torch::ScalarType get_scalar_type(ScalarTypeEnum type_enum) {
    switch (type_enum) {
        case ScalarTypeEnum::float16:
            return torch::kFloat16;
        case ScalarTypeEnum::float32:
            return torch::kFloat32;
        case ScalarTypeEnum::bfloat16:
            return torch::kBFloat16;
        default:
            throw py::type_error("Unsupported scalar type");
    }
};

torch::Tensor dequantize_fp4(torch::Tensor A, torch::Tensor absmax, int blocksize, int M, int N, ScalarTypeEnum o_type) {
    CHECK_CUDA(A);
    CHECK_CUDA(absmax);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(absmax);

    torch::Tensor out = torch::empty({M, N}, torch::TensorOptions().dtype(get_scalar_type(o_type)).device(A.device()));
    dequantize_blockwise_fp4(A, absmax, M, N, blocksize, M * N, out);
    return out;
}

torch::Tensor dequantize_fp4_codebook(
    torch::Tensor A, torch::Tensor absmax, torch::Tensor codebook, int M, int N, int blocksize, int n, ScalarTypeEnum dtype
) {
    CHECK_CUDA(A);
    CHECK_CUDA(absmax);
    CHECK_CUDA(codebook);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(absmax);
    CHECK_CONTIGUOUS(codebook);
    return dequantize_blockwise_codebook_fp4(A, absmax, codebook, M, N, blocksize, n, get_scalar_type(dtype));
}

torch::Tensor qlinear(torch::Tensor A_in, torch::Tensor A, torch::Tensor absmax, int M, int N, int blocksize) {
    CHECK_CUDA(A);
    CHECK_CUDA(absmax);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(absmax);
    torch::Tensor out = torch::empty({M, N}, A_in.options());
    dequantize_blockwise_fp4(A, absmax, M, N, blocksize, M * N, out);
    return torch::nn::functional::linear(A_in, out);
}

torch::Tensor qlinear_bias(torch::Tensor A_in, torch::Tensor A, torch::Tensor absmax, int M, int N, int blocksize, torch::Tensor bias) {
    CHECK_CUDA(A);
    CHECK_CUDA(absmax);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(absmax);
    torch::Tensor out = torch::empty({M, N}, A_in.options());
    dequantize_blockwise_fp4(A, absmax, M, N, blocksize, M * N, out);
    return torch::nn::functional::linear(A_in, out, bias);
}

torch::Tensor
qlinear_codebook(torch::Tensor A_in, torch::Tensor A, torch::Tensor absmax, torch::Tensor codebook, int M, int N, int blocksize) {
    CHECK_CUDA(A);
    CHECK_CUDA(absmax);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(absmax);
    torch::Tensor weight = dequantize_blockwise_codebook_fp4(A, absmax, codebook, M, N, blocksize, A.numel(), A_in.scalar_type());
    return torch::nn::functional::linear(A_in, weight);
}

torch::Tensor qlinear_codebook_bias(
    torch::Tensor A_in, torch::Tensor A, torch::Tensor absmax, torch::Tensor codebook, int M, int N, int blocksize, torch::Tensor bias
) {
    CHECK_CUDA(A);
    CHECK_CUDA(absmax);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(absmax);
    torch::Tensor weight = dequantize_blockwise_codebook_fp4(A, absmax, codebook, M, N, blocksize, A.numel(), A_in.scalar_type());
    return torch::nn::functional::linear(A_in, weight, bias);
}

torch::Tensor gemv_fp4(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor absmax,
    torch::Tensor datatype,
    int blocksize,
    ScalarTypeEnum dtype,
    std::vector<uint32_t> Bshape
) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_CUDA(absmax);
    CHECK_CUDA(datatype);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);
    CHECK_CONTIGUOUS(absmax);
    CHECK_CONTIGUOUS(datatype);
    return gemv_4bit_inference(A, B, absmax, datatype, blocksize, get_scalar_type(dtype), Bshape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::enum_<ScalarTypeEnum>(m, "ScalarType")
        .value("bfloat16", ScalarTypeEnum::bfloat16)
        .value("float16", ScalarTypeEnum::float16)
        .value("float32", ScalarTypeEnum::float32)
        .export_values();

    m.def("dequantize_fp4", &dequantize_fp4, "A test function for dequantize_fp4");
    m.def("dequantize_fp4_codebook", &dequantize_fp4_codebook, "A test function for dequantize_fp4_interface");
    m.def("gemv_fp4", &gemv_fp4, "A test function for gemm_4bit_inference_impl");
    m.def("qlinear", &qlinear, "A test function for qlinear");
    m.def("qlinear_bias", &qlinear_bias, "A test function for qlinear with bias");
    m.def("qlinear_codebook", &qlinear_codebook, "A test function for qlinear with codebook");
    m.def("qlinear_codebook_bias", &qlinear_codebook_bias, "A test function for qlinear with codebook and bias");
}