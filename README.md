# TORCH BNB FP4

torch_bnb_fp4 is a library that provides a Torch C++ extension for faster nn.Linear FP4 ops, via streamlining bitsandbytes [`kgemm_4bit_inference_naive`](https://github.com/TimDettmers/bitsandbytes/blob/e820409c095ea7cbb5ce156992307b84352cbf90/csrc/kernels.cu#L3533-L3649) and [`kDequantizeBlockwise`](https://github.com/TimDettmers/bitsandbytes/blob/e820409c095ea7cbb5ce156992307b84352cbf90/csrc/kernels.cu#L832-L896) kernels.

## Overview

TORCH BNB FP4 is a high-performance library designed to accelerate quantized `nn.Linear` ops, by utilizing bitsandbytes fp4 quantized weights. This library is built as a Torch C++ extension instead of being linked via ctypes as with bitsandbytes. This library is designed to be used in conjunction with bitsandbytes, and is not a replacement for bitsandbytes.

## Requirements

System:

- CUDA capable device with compute >= 8.0, so only Ampere / Ada / Hopper and above.
- [System cudatoolkit](https://developer.nvidia.com/cuda-downloads) with the same major version eg 11.x, 12.x as their installed pytorch's cuda. Minor version mismatches dont matter as much, as in, 12.1 pytorch will work fine with system cudatoolkit 12.3, etc. This is specifically for the libs & headers of NVIDIA CUB.

Note:

- _I am 100% unsure whether this works on (non-wsl) windows at all._
- I have only tested this on a 4090 on linux with cudatoolkit 12.3 w/ pytorch2.2+cuda=12.1, a 4080 with cudatoolkit 12.2 & pytorch2.2+cuda=12.1 on windows w/ wsl, and a 3090 on linux with cudatoolkit 12.2 & pytorch2.2+cuda=12.1. Other setups are not guaranteed to work, but only because I have not tested them. If you find issues, feel welcome to submit an issue with your cudatoolkit version, cuda device and the errors you had.

Libraries:

- Pytorch
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

## Installation

To install torch_bnb_fp4, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/aredden/torch-bnb-fp4
   ```

2. Navigate to the project directory:

   ```bash
   cd torch-bnb-fp4
   ```

3. To reduce the chance of issues finding the correct libraries / headers, I recommend finding your cuda library directory and referencing them in the install command, since frequently your PATH env variable ends up overwriting your system cudatoolkit library / include dirs with older cudatoolkit installations.

   - You will need to specify the actual compute architecture in the TORCH_CUDA_ARCH_LIST environment variable, Ampere consumer gpus are 8.6, Ada rtx 40xx gpus and workstation cards are 8.9, and hopper datacenter gpus are 9.0.

     - For an ampere A100 I would use `TORCH_CUDA_ARCH_LIST="8.0"`
       - ampere datacenter cards are a special case for ampere, I am unsure whether all are 8.0 or just the A100, so be sure to check.
     - For an ampere RTX 3070 I would use `TORCH_CUDA_ARCH_LIST="8.6"`
     - For an ada RTX 4080 I would use `TORCH_CUDA_ARCH_LIST="8.9"`
     - For a hopper H100 I would use `TORCH_CUDA_ARCH_LIST="9.0"`
     - ...

   - On linux and wsl, the library directory it is usually `/usr/local/cuda-x.y/lib64`, and the nvcc nvidia compiler is usually `/usr/local/cuda-x.y/bin/nvcc`, where `x` is the cudatoolkit major version, and `y` is the minor version, eg for cudatoolkit 12.2, you would use `/usr/local/cuda-12.2/lib64` then you can use the install command:

     ```bash
     # assuming cudatoolkit 12.2 and cuda your device is a 3090 (aka compute 8.6)

     export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64
     export CUDACXX=/usr/local/cuda-12.2/bin/nvcc
     export CUDA_HOME=/usr/local/cuda-12.2
     TORCH_CUDA_ARCH_LIST="8.6" python setup.py install
     ```

     OR, if you're feeling lucky / know your system has all libs / headers properly set up:

     ```bash
     # assuming your device is a 4090 (aka compute 8.9)

     TORCH_CUDA_ARCH_LIST="8.9" python setup.py install
     ```

## Usage

Once the library is installed, you can use it in your Torch projects by importing the `torch_bnb_fp4` module, which provides access to the pytorch extension.

To make sure things are working correctly, you can use the script `sanity_check.py` in the root of this repository, which tests the speed and accuracy of the library. For reference, the output from my gpu is as follows:

```

❯ python sanity_check.py

============ Running Sanity Checks ============


 NOTE: The acceptable range for the elementwise difference avg
 is around 0.045-0.065, which is the same as bitsandbytes.

== Running sanity check for torch-bnb-fp4 fp32 ==

+------------------------------------------------------------------------------+
| GEMM Speed Benchmark for torch.float32 and matmul type [GEMV] W/ 6 Layer MLP |
+-----------------------+----------------+-------------------+-----------------+
|          type         |   mean (us)    |    median (us)    |     iqr (us)    |
+-----------------------+----------------+-------------------+-----------------+
|        pytorch        |    53.18113    |      53.09262     |     0.12039     |
|      bitsandbytes     |    92.71299    |      92.70629     |     0.16016     |
|     torch-bnb-fp4     |    63.77637    |      63.78534     |      0.0904     |
+-----------------------+----------------+-------------------+-----------------+
+------------------------------------------------------------------------------+
| GEMM Speed Benchmark for torch.float32 and matmul type [GEMM] W/ 6 Layer MLP |
+-----------------------+----------------+-------------------+-----------------+
|          type         |   mean (us)    |    median (us)    |     iqr (us)    |
+-----------------------+----------------+-------------------+-----------------+
|        pytorch        |    68.58508    |      68.58716     |     0.02236     |
|      bitsandbytes     |   155.64296    |     155.13446     |     1.37504     |
|     torch-bnb-fp4     |    93.45283    |      93.4459      |     0.02174     |
+-----------------------+----------------+-------------------+-----------------+
Elementwise Diff. Avg Between nn.Linear & Quant GEMV 3dim: 0.05073589086532593
Elementwise Diff. Avg Between nn.Linear & Quant GEMV 2dim: 0.056356318295001984
Elementwise Diff. Avg Between nn.Linear & Quant GEMM 3dim: 0.05096859857439995

== Running sanity check for torch-bnb-fp4 fp16 ==

+------------------------------------------------------------------------------+
| GEMM Speed Benchmark for torch.float16 and matmul type [GEMV] W/ 6 Layer MLP |
+-----------------------+----------------+-------------------+-----------------+
|          type         |   mean (us)    |    median (us)    |     iqr (us)    |
+-----------------------+----------------+-------------------+-----------------+
|        pytorch        |    54.0681     |      53.92455     |     0.28024     |
|      bitsandbytes     |    93.89957    |      93.93588     |     0.22058     |
|     torch-bnb-fp4     |    64.42346    |      64.4473      |     0.04361     |
+-----------------------+----------------+-------------------+-----------------+
+------------------------------------------------------------------------------+
| GEMM Speed Benchmark for torch.float16 and matmul type [GEMM] W/ 6 Layer MLP |
+-----------------------+----------------+-------------------+-----------------+
|          type         |   mean (us)    |    median (us)    |     iqr (us)    |
+-----------------------+----------------+-------------------+-----------------+
|        pytorch        |    79.42544    |      79.41179     |      0.0154     |
|      bitsandbytes     |   130.14084    |      130.1941     |     0.54197     |
|     torch-bnb-fp4     |    98.83817    |      98.83849     |      0.0185     |
+-----------------------+----------------+-------------------+-----------------+
Elementwise Diff. Avg Between nn.Linear & Quant GEMV 3dim: 0.04998779296875
Elementwise Diff. Avg Between nn.Linear & Quant GEMV 2dim: 0.05657958984375
Elementwise Diff. Avg Between nn.Linear & Quant GEMM 3dim: 0.05096435546875

== Running sanity check for torch-bnb-fp4 bf16 ==

+-------------------------------------------------------------------------------+
| GEMM Speed Benchmark for torch.bfloat16 and matmul type [GEMV] W/ 6 Layer MLP |
+-----------------------+----------------+--------------------+-----------------+
|          type         |   mean (us)    |    median (us)     |     iqr (us)    |
+-----------------------+----------------+--------------------+-----------------+
|        pytorch        |    54.3889     |      54.14199      |     0.39099     |
|      bitsandbytes     |    94.2237     |      93.96561      |     0.60638     |
|     torch-bnb-fp4     |    64.3852     |      64.35706      |     0.21559     |
+-----------------------+----------------+--------------------+-----------------+
+-------------------------------------------------------------------------------+
| GEMM Speed Benchmark for torch.bfloat16 and matmul type [GEMM] W/ 6 Layer MLP |
+-----------------------+----------------+--------------------+-----------------+
|          type         |   mean (us)    |    median (us)     |     iqr (us)    |
+-----------------------+----------------+--------------------+-----------------+
|        pytorch        |    81.96011    |      81.94626      |     0.01879     |
|      bitsandbytes     |   152.93054    |     152.84844      |     0.50242     |
|     torch-bnb-fp4     |   101.29481    |     101.28148      |     0.02136     |
+-----------------------+----------------+--------------------+-----------------+
Elementwise Diff. Avg Between nn.Linear & Quant GEMV 3dim: 0.049072265625
Elementwise Diff. Avg Between nn.Linear & Quant GEMV 2dim: 0.05712890625
Elementwise Diff. Avg Between nn.Linear & Quant GEMM 3dim: 0.051025390625

============= Sanity Checks Compete =============

```

The library provides a `TorchFP4Linear` class that can be used to replace standard PyTorch nn.Linear layers via bitsandbytes FP4 quantized layers.

```py
from torch import nn
from torch_bnb_fp4 import TorchFP4Linear, swap_linear_with_bnb_linear

# Define your original linear layer
# NOTE: this lib supports float16, bfloat16 and float32 tensors.
original_linear_layer = nn.Linear(
    in_features=512,
    out_features=1024,
    bias=True
).to(device='cuda', dtype=torch.float16)

original_linear_layer = swap_linear_with_bnb_linear(
    original_linear_layer,
    dtype=torch.float16
).cuda() # cuda must be called to quantize the linear weights via bnb.

# wrap the linear layer via passing to the constructor of the TorchFP4Linear layer.
quantized_linear_layer = TorchFP4Linear(
    original_linear_layer,
    use_codebook_dequant=True # or False for fp4 tree dequant, though doesn't make much difference.
).to(device='cuda', dtype=torch.float16)

# Use the quantized layer as you would with a standard nn.Linear layer
input_tensor = torch.randn(10, 512).to(device='cuda', dtype=torch.float16)
output = quantized_linear_layer(input_tensor)

# output is now a torch.float16 tensor, and can be used as input to other torch-bnb-fp4'd layers or models.

```

For huggingface models, I recommend loading as bitsandbytes fp4 quantized model, and then recursively replacing the BNB layers with the TorchFP4Linear layers.

```py
import torch
from torch_bnb_fp4 import recursively_replace_with_fp4_linear
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Change this to your desired dtype
DTYPE = torch.float16

# Load weights as bnb fp4
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map={"": 0},
    torch_dtype=DTYPE,
    load_in_4bit=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=DTYPE,
        # Must use "fp4" for this library
        bnb_4bit_quant_type="fp4",
    )
)

# Replace layers with torch-bnb-fp4 layers in-place
model = recursively_replace_with_fp4_linear(
    model,
    as_dtype=DTYPE,
    use_codebook_dequant=True # or False for fp4 tree dequant, though doesn't make much difference.
)

# Now your model is torch-bnb-fp4'd



```

## Acknowledgements

I would like to thank Tim Dettmers for creating bitsandbytes and providing 99.99% of the foundation for this library. For more detailed information on the underlying quantization techniques, refer to the [bitsandbytes GitHub repository](https://github.com/TimDettmers/bitsandbytes).
