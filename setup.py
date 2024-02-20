from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import find_packages

flags = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
]


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]


setup(
    name="torch_bnb_fp4",
    version="0.0.2",
    packages=find_packages(exclude=['csrc','csrc/*']),
    requires=['bitsandbytes'],
    ext_modules=[
        CUDAExtension(
            name="torch_bnb_fp4_ext",
            sources=[
                "csrc/gemv_fp4_optimized.cu",
                "csrc/dequant_fp4_optimized.cu",
                "csrc/torch_fp4.cpp",
            ],
            extra_compile_args={
                "cxx": ["-g", "-O3", "-std=c++17"],
                "nvcc": flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
