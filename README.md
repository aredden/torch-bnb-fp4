# torch_bnb_fp4

torch_bnb_fp4 is a library that provides a Torch C++ extension for bitsandbytes dequantization. It allows for efficient dequantization of FP4 (4 bit floating-point) values.

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

3. Replace the `TORCH_CUDA_ARCH_LIST` flag in the `install.bash` with your CUDA architecture flag. For example, if your CUDA architecture is `8.0`, modify the `install.bash` script as follows:

    ```bash
    TORCH_CUDA_ARCH_LIST="8.0" python setup.py install
    ```

> NOTE: This library will likely only work with torch architectures >= 8.0, since I removed a lot of the if/else conditionals in the kernels for slight performance improvements. 

5. Build and install the library:

    ```bash
    bash ./install.bash
    ```

## Usage

Once the library is installed, you can use it in your Torch projects by importing the `torch_bnb_fp4` module, which provides access to the pytorch extension.