import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    _get_cuda_arch_flags,
)


def check_device_capability_minimum_allowed():
    if os.environ.get("IGNORE_DEVICE_CHECK", "0") == "1":
        print("Ignoring device check (intended for docker builds / CI)")
    else:
        count = torch.cuda.device_count()
        for c in range(count):
            i = torch.cuda.get_device_capability(c)
            print(f"Device {c}: {i}")
            if i[0] < 8:
                raise ValueError(
                    "Minimum compute capability is 80, if you are compiling this extension without a device, such as in a docker container or CI, set the IGNORE_DEVICE_CHECK environment variable to 1 to ignore this check"
                )
        if os.getenv("TORCH_CUDA_ARCH_LIST", "") != "":
            archs = _get_cuda_arch_flags()
            archs = [int(x.rsplit("_", 1)[-1]) for x in archs]
            for arch in archs:
                if arch < 80:
                    raise ValueError(
                        "Minimum compute capability is 80, if you are compiling this extension without a device, such as in a docker container or CI, set the IGNORE_DEVICE_CHECK environment variable to 1 to ignore this check"
                    )


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


# Make sure the device is capable
check_device_capability_minimum_allowed()

setup(
    name="torch_bnb_fp4",
    version="0.0.4",
    packages=find_packages(
        exclude=[
            "csrc",
            "csrc/*",
            ".misc",
            ".misc/*",
        ]
    ),
    requires=["bitsandbytes", "prettytable"],
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
