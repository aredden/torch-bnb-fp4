#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/Exceptions.h>


void dequantizeBlockwise_impl(torch::Tensor A, torch::Tensor absmax, int M, int N, int blocksize, int n, torch::Tensor out);
torch::Tensor gemv_4bit_inference(torch::Tensor A, torch::Tensor B, torch::Tensor absmax, torch::Tensor datatype, int blocksize, torch::ScalarType dtype, std::vector<uint32_t> Bshape, bool use_reduced_prec_accumulate);


#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), " must be contiguous")

enum class ScalarTypeEnum
{
    float16,
    float32,
    bfloat16,
};

torch::ScalarType get_scalar_type(ScalarTypeEnum type_enum)
{
    switch (type_enum)
    {
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

torch::Tensor dequantize_fp4(torch::Tensor A, torch::Tensor absmax,
                             int blocksize, int M, int N, ScalarTypeEnum o_type)
{
    CHECK_CUDA(A);
    CHECK_CUDA(absmax);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(absmax);
    
    torch::Tensor out = torch::empty({M, N}, torch::TensorOptions().dtype(get_scalar_type(o_type)).device(A.device()));
    dequantizeBlockwise_impl(A, absmax, M, N, blocksize, M * N, out);
    return out;
}

torch::Tensor qlinear(torch::Tensor A_in, torch::Tensor A, torch::Tensor absmax, int M, int N, int blocksize)
{
    CHECK_CUDA(A);
    CHECK_CUDA(absmax);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(absmax);
    torch::Tensor out = torch::empty({M, N}, A_in.options());
    dequantizeBlockwise_impl(A, absmax, M, N, blocksize, M * N, out);
    return torch::nn::functional::linear(A_in, out);
}

torch::Tensor qlinear_bias(torch::Tensor A_in, torch::Tensor A, torch::Tensor absmax, int M, int N, int blocksize, torch::Tensor bias)
{
    CHECK_CUDA(A);
    CHECK_CUDA(absmax);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(absmax);
    torch::Tensor out = torch::empty({M, N}, A_in.options());
    dequantizeBlockwise_impl(A, absmax, M, N, blocksize, M * N, out);
    return torch::nn::functional::linear(A_in, out, bias);
}

torch::Tensor gemv_4bit_inference_impl(torch::Tensor A, torch::Tensor B, torch::Tensor absmax, torch::Tensor datatype, int blocksize, ScalarTypeEnum dtype, std::vector<uint32_t> Bshape)
{
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_CUDA(absmax);
    CHECK_CUDA(datatype);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);
    CHECK_CONTIGUOUS(absmax);
    CHECK_CONTIGUOUS(datatype);
    return gemv_4bit_inference(A, B, absmax, datatype, blocksize, get_scalar_type(dtype), Bshape, false);
}

torch::Tensor gemv_4bit_inference_impl_reduced_prec(torch::Tensor A, torch::Tensor B, torch::Tensor absmax, torch::Tensor datatype, int blocksize, ScalarTypeEnum dtype, std::vector<uint32_t> Bshape)
{
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_CUDA(absmax);
    CHECK_CUDA(datatype);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);
    CHECK_CONTIGUOUS(absmax);
    CHECK_CONTIGUOUS(datatype);
    return gemv_4bit_inference(A, B, absmax, datatype, blocksize, get_scalar_type(dtype), Bshape, true);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::enum_<ScalarTypeEnum>(m, "ScalarType")
        .value("bfloat16", ScalarTypeEnum::bfloat16)
        .value("float16", ScalarTypeEnum::float16)
        .value("float32", ScalarTypeEnum::float32)
        .export_values();

    m.def("dequantize_fp4", &dequantize_fp4,
          "A test function for dequantize_fp4");
    m.def("gemv_4bit_inference_impl", &gemv_4bit_inference_impl,
          "A test function for gemm_4bit_inference_impl");
    m.def("gemv_4bit_inference_impl_reduced_prec", &gemv_4bit_inference_impl_reduced_prec,
          "A test function for gemm_4bit_inference_impl_reduced_prec");
    m.def("qlinear", &qlinear,
            "A test function for qlinear");
    m.def("qlinear_bias", &qlinear_bias,
            "A test function for qlinear with bias");

}