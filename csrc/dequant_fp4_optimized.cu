#include <ATen/cuda/CUDAContext.h>
#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>
#include <cub/cub.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits.h>
#include <math_constants.h>
#include <mma.h>
#include <stdio.h>
#include <vector>
#define HLF_MAX 65504
#define TH 1024
#define NUM 4
#define NUM_BLOCK 4096
#define CDIV(x, y) (((x) + (y)-1) / (y))

void CUDA_CHECK_RETURN_(cudaError_t cudaStatus) {
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Failure: %s\n", cudaGetErrorString(cudaStatus));
        // exit(EXIT_FAILURE); // so many segfaults before being able to print out actual crap because of this stupidity
    }
}

__device__ float dequantize_fp4_tree(unsigned char val, float absmax) {
    float sign = (val & 0b1000) == 8 ? -1.0f : 1.0f;
    if ((val & 0b0100) == 4) // 0
        if ((val & 0b0010) == 2) // 01
            if ((val & 0b0001) == 1) // 111
                return 0.25000000f * absmax * sign; // 1111
            else
                return 0.16666667f * absmax * sign; // 1110
        else if ((val & 0b0001) == 1) // 110
            return 0.50000000f * absmax * sign; // 1101
        else
            return 0.33333333f * absmax * sign; // 1100
    else if ((val & 0b0010) == 2) // 10
        if ((val & 0b0001) == 1) // 101
            return 1.00000000f * absmax * sign; // 1011
        else
            return 0.66666667f * absmax * sign; // 1010
    else if ((val & 0b0001) == 1) // 100
        return 5.208333333e-03f * absmax * sign; // 1001
    else
        return 0.00000000f * absmax * sign; // 1000
}

template <typename T> __device__ __forceinline__ T convert_to_ty(float val);
template <> __device__ __forceinline__ nv_bfloat16 convert_to_ty(float val) {
    return __float2bfloat16(val);
}
template <> __device__ __forceinline__ nv_half convert_to_ty(float val) {

    return __float2half(val);
}
template <> __device__ __forceinline__ float convert_to_ty(float val) {
    return val;
}

template <typename T, int TILE_SIZE, int THREADS, int NUM_PER_TH>
__global__ void
dequantize_blockwise_kernel_cuda(unsigned char *A, float *absmax, T *out, const int blocksize, const int n) {
    const int n_load = (gridDim.x * TILE_SIZE);
    int valid_items_load;
    int valid_items_store;
    const int base_idx = (blockIdx.x * TILE_SIZE);
    T vals[NUM_PER_TH * 2];
    unsigned char qvals[NUM_PER_TH];
    float local_abs_max;

    valid_items_load = 0;
    valid_items_store = 0;
    local_abs_max = -FLT_MAX;
    typedef cub::BlockLoad<unsigned char, THREADS, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;
    typedef cub::BlockStore<T, THREADS, NUM_PER_TH * 2, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

    __shared__ typename LoadChar::TempStorage loadchar;
    __shared__ typename StoreT::TempStorage storet;
    for (unsigned int i = base_idx; i < n_load; i += gridDim.x * TILE_SIZE) {
        valid_items_load = (n + 1) / 2 - i > TILE_SIZE ? TILE_SIZE : (n + 1) / 2 - i;
        valid_items_store = n - i * 2 > TILE_SIZE * 2 ? TILE_SIZE * 2 : n - i * 2;
        local_abs_max = __ldg(&absmax[(i + threadIdx.x * NUM_PER_TH) / (blocksize)]);
        __syncthreads();

        LoadChar(loadchar).Load(&(A[i]), qvals, valid_items_load, 128);
#pragma unroll NUM_PER_TH
        for (int j = 0; j < NUM_PER_TH; j++) {

            vals[j * 2] = convert_to_ty<T>(dequantize_fp4_tree(qvals[j] >> 4, local_abs_max));
            vals[j * 2 + 1] = convert_to_ty<T>(dequantize_fp4_tree(qvals[j] & 0x0F, local_abs_max));
        }
        __syncthreads();
        StoreT(storet).Store(&(out[i * 2]), vals, valid_items_store);
    }
}

template <typename T>
void launch_kernel_dequant(torch::Tensor A, torch::Tensor absmax, torch::Tensor out, int blocksize, int n) {

    dequantize_blockwise_kernel_cuda<T, 512, 64, 8><<<(n + 1024 - 1) / 1024, 64>>>(
        (unsigned char *)A.data_ptr(),
        (float *)absmax.data_ptr(),
        (T *)out.mutable_data_ptr(),
        (const int)(blocksize / 2),
        (const int)n
    );
    CUDA_CHECK_RETURN_(cudaGetLastError());
}

template <typename T>
void launch_kernel_dequant_stream(
    torch::Tensor A, torch::Tensor absmax, torch::Tensor out, int blocksize, int n, cudaStream_t stream
) {

    dequantize_blockwise_kernel_cuda<T, 512, 64, 8><<<(n + 1024 - 1) / 1024, 64, 0, stream>>>(
        (unsigned char *)A.data_ptr(),
        (float *)absmax.data_ptr(),
        (T *)out.mutable_data_ptr(),
        (const int)(blocksize / 2),
        (const int)n
    );
    CUDA_CHECK_RETURN_(cudaGetLastError());
}

void dequantizeBlockwise_impl(
    torch::Tensor A, torch::Tensor absmax, int M, int N, int blocksize, int n, torch::Tensor out
) {
    TORCH_CHECK(A.dtype() == torch::kUInt8, "A must be uint8");
    TORCH_CHECK(absmax.dtype() == torch::kFloat32, "absmax must be float32");
    TORCH_CHECK(A.is_cuda(), "A must be cuda");
    TORCH_CHECK(absmax.is_cuda(), "absmax must be cuda");
    TORCH_CHECK(out.is_cuda(), "out must be cuda");
    switch (out.scalar_type()) {
        case torch::kFloat16: {
            launch_kernel_dequant<nv_half>(A, absmax, out, blocksize, n);
            break;
        }
        case torch::kFloat32: {
            launch_kernel_dequant<float>(A, absmax, out, blocksize, n);
            break;
        }
        case torch::kBFloat16: {
            launch_kernel_dequant<nv_bfloat16>(A, absmax, out, blocksize, n);
            break;
        }
        default: {
            std::cout << "NO APPLICABLE DEQUANT DTYPE!" << std::endl;
        }
    }
}

void dequantizeBlockwise_impl_stream(
    torch::Tensor A, torch::Tensor absmax, int M, int N, int blocksize, int n, torch::Tensor out, cudaStream_t stream
) {
    TORCH_CHECK(A.dtype() == torch::kUInt8, "A must be uint8");
    TORCH_CHECK(absmax.dtype() == torch::kFloat32, "absmax must be float32");
    TORCH_CHECK(A.is_cuda(), "A must be cuda");
    TORCH_CHECK(absmax.is_cuda(), "absmax must be cuda");
    TORCH_CHECK(out.is_cuda(), "out must be cuda");

    switch (out.scalar_type()) {
        case torch::kFloat16: {
            launch_kernel_dequant_stream<nv_half>(A, absmax, out, blocksize, n, stream);
            break;
        }
        case torch::kFloat32: {
            launch_kernel_dequant_stream<float>(A, absmax, out, blocksize, n, stream);
            break;
        }
        case torch::kBFloat16: {
            launch_kernel_dequant_stream<nv_bfloat16>(A, absmax, out, blocksize, n, stream);
            break;
        }
        default: {
            std::cout << "NO APPLICABLE DEQUANT DTYPE!" << std::endl;
        }
    }
}
