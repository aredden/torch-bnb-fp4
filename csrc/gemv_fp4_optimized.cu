#include <ATen/cuda/CUDAContext.h>
#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>
#include <cub/cub.cuh>
#include <cub/util_math.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <torch/extension.h>
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
#define num_values_4bit 32
#define num_values_4bit 32

typedef struct {
    float param[16];
} param_large_t;

static const param_large_t CODE_PARAM = {
    .param =
        {0.00000f,
         5.208333e-03f,
         0.6666667f,
         1.000000f,
         0.333333f,
         0.500000f,
         0.1666667f,
         0.250000f,
         -0.000000f,
         -5.208333e-03f,
         -0.6666667f,
         -1.000000f,
         -0.333333f,
         -0.500000f,
         -0.1666667f,
         -0.250000f}
};

#define CDIV(x, y) (((x) + (y)-1) / (y))

void CUDA_CHECK_RETURN(cudaError_t cudaStatus) {
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Failure: %s\n", cudaGetErrorString(cudaStatus));
    }
}

template <typename T, int THREADS, typename T_REDUCE>
__global__ void gemv_4bit_inference_kernel(
    int M,
    int N,
    int K,
    T *__restrict__ const A,
    unsigned char *B,
    T_REDUCE *absmax,
    __grid_constant__ const param_large_t datatype,
    T *out,
    int lda,
    int ldb,
    int ldc,
    int blocksize
) {
    // per threadblock:
    // load step-by-step in chunks of [32,warps]: 1x32 * [32,warps] -> [1,warps]
    // 4 warps -> 4 loads per iter
    // 1x32 * 32x4 -> 1x4 outputs per thread block

    typedef cub::WarpReduce<T_REDUCE> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[THREADS / 32];

    const int warp_idx = threadIdx.x / 32;
    const int warp_lane = threadIdx.x % 32;
    const int row_B = (THREADS / 32) * blockIdx.x + warp_idx;
    const int num_values_8bit = num_values_4bit / 2;
    T local_C = T(0.0f);

    unsigned char local_B_4bit[num_values_8bit];
    T local_B[num_values_4bit / 4];
    T local_A[num_values_4bit / 4];
    __shared__ T quant_map[16];
    T local_absmax = T(0.0f);

    if (warp_lane < 16 && warp_idx == 0) quant_map[warp_lane] = T(datatype.param[warp_lane]);
    __syncthreads();
    // A: [1, K]
    // B: [N, K]
    for (int inner_idx = warp_lane * num_values_4bit; inner_idx < K; inner_idx += 32 * num_values_4bit) {
        int inner_idx_halved = inner_idx / 2;
        int offset_B = ldb * row_B;
        int absidx = ((2 * offset_B) + inner_idx) / blocksize;
        local_absmax = __ldg(&(absmax[absidx]));

        if (row_B < M) {
            if ((inner_idx_halved + num_values_8bit) < (K / 2)) {
                reinterpret_cast<int4(&)[num_values_8bit]>(local_B_4bit)[0] =
                    reinterpret_cast<int4 *>(B)[(offset_B + (inner_idx_halved)) / (num_values_8bit)];
            } else {
#pragma unroll
                for (int j = 0; j < (num_values_8bit); j++) {
                    if ((inner_idx_halved) + j < (K / 2)) {
                        local_B_4bit[j] = B[offset_B + inner_idx_halved + j];
                    } else {
                        local_B_4bit[j] = 0b01110111;
                    }
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < (num_values_8bit); j++) {
                local_B_4bit[j] = 0b01110111;
            }
        }
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int k = 0; k < num_values_8bit / 4; k++) {
                local_B[k * 2] = quant_map[local_B_4bit[(i * num_values_8bit / 4) + k] >> 4] * local_absmax;
                local_B[k * 2 + 1] = quant_map[local_B_4bit[(i * num_values_8bit / 4) + k] & 0x0F] * local_absmax;
            }
            if (inner_idx + (num_values_4bit / 4) + (i * num_values_4bit / 4) < K) {
                // this is also relatively important for performance
                reinterpret_cast<int4(&)[num_values_4bit]>(local_A)[0] = reinterpret_cast<int4 *>(A)[inner_idx / (num_values_4bit / 4) + i];
            } else {
#pragma unroll
                for (int k = 0; k < num_values_4bit / 4; k++) {
                    if (inner_idx + (i * num_values_4bit / 4) + k < K) {
                        local_A[k] = A[inner_idx + k + (i * num_values_4bit / 4)];
                    } else {
                        local_A[k] = T(0.0f);
                    }
                }
            }
            // accumulate in float; small performance hit for Ampere, but lower error for outputs
#pragma unroll
            for (int k = 0; k < num_values_4bit / 4; k++) {
                local_C += local_A[k] * local_B[k];
            }
        }
    }

    local_C = WarpReduce(temp_storage[warp_idx]).Sum(local_C);

    if (row_B < M && warp_lane == 0) {
        out[row_B] = T(local_C);
    };
}

template <typename T, int THREADS, typename T_REDUCE>
__global__ void gemv_4bit_inference_kernel_float(
    int M,
    int N,
    int K,
    T *__restrict__ const A,
    unsigned char *B,
    T_REDUCE *absmax,
    __grid_constant__ const param_large_t datatype,
    T *out,
    int lda,
    int ldb,
    int ldc,
    int blocksize
) {
    // per threadblock:
    // load step-by-step in chunks of [32,warps]: 1x32 * [32,warps] -> [1,warps]
    // 4 warps -> 4 loads per iter
    // 1x32 * 32x4 -> 1x4 outputs per thread block

    typedef cub::WarpReduce<T_REDUCE> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[THREADS / 32];

    const int warp_idx = threadIdx.x / 32;
    const int warp_lane = threadIdx.x % 32;
    const int row_B = (THREADS / 32) * blockIdx.x + warp_idx;
    const int num_values_8bit = num_values_4bit / 2;
    T local_C = T(0.0f);

    unsigned char local_B_4bit[num_values_8bit];
    T local_B[num_values_4bit / 4];
    T local_A[num_values_4bit / 4];
    __shared__ T quant_map[16];
    T local_absmax = T(0.0f);

    if (warp_lane < 16 && warp_idx == 0) quant_map[warp_lane] = T(datatype.param[warp_lane]);
    __syncthreads();
    // A: [1, K]
    // B: [N, K]
    for (int inner_idx = warp_lane * num_values_4bit; inner_idx < K; inner_idx += 32 * num_values_4bit) {
        int inner_idx_halved = inner_idx / 2;
        int offset_B = ldb * row_B;
        int absidx = ((2 * offset_B) + inner_idx) / blocksize;
        local_absmax = __ldg(&(absmax[absidx]));

        if (row_B < M) {
            if ((inner_idx_halved + num_values_8bit) < (K / 2)) {
                reinterpret_cast<int4(&)[num_values_8bit]>(local_B_4bit)[0] =
                    reinterpret_cast<int4 *>(B)[(offset_B + (inner_idx_halved)) / (num_values_8bit)];
            } else {
#pragma unroll
                for (int j = 0; j < (num_values_8bit); j++) {
                    if ((inner_idx_halved) + j < (K / 2)) {
                        local_B_4bit[j] = B[offset_B + inner_idx_halved + j];
                    } else {
                        local_B_4bit[j] = 0b01110111;
                    }
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < (num_values_8bit); j++) {
                local_B_4bit[j] = 0b01110111;
            }
        }
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int k = 0; k < num_values_8bit / 4; k++) {
                local_B[k * 2] = quant_map[local_B_4bit[(i * num_values_8bit / 4) + k] >> 4] * local_absmax;
                local_B[k * 2 + 1] = quant_map[local_B_4bit[(i * num_values_8bit / 4) + k] & 0x0F] * local_absmax;
            }
            if (inner_idx + (num_values_4bit / 4) + (i * num_values_4bit / 4) < K) {
                // this is also relatively important for performance
                reinterpret_cast<int4(&)[num_values_4bit]>(local_A)[0] =
                    reinterpret_cast<int4 *>(A)[inner_idx / (num_values_4bit / 8) + (2 * i) + 0];
                reinterpret_cast<int4(&)[num_values_4bit]>(local_A)[1] =
                    reinterpret_cast<int4 *>(A)[inner_idx / (num_values_4bit / 8) + (2 * i) + 1];
            } else {
#pragma unroll
                for (int k = 0; k < num_values_4bit / 4; k++) {
                    if (inner_idx + (i * num_values_4bit / 4) + k < K) {
                        local_A[k] = A[inner_idx + k + (i * num_values_4bit / 4)];
                    } else {
                        local_A[k] = T(0.0f);
                    }
                }
            }
            // accumulate in float; small performance hit for Ampere, but lower error for outputs
#pragma unroll
            for (int k = 0; k < num_values_4bit / 4; k++) {
                local_C += local_A[k] * local_B[k];
            }
        }
    }

    local_C = WarpReduce(temp_storage[warp_idx]).Sum(local_C);

    if (row_B < M && warp_lane == 0) {
        out[row_B] = T(local_C);
    };
}

template <typename T, typename T_REDUCE>
void gemv_4bit_inference_launch(
    int m, int n, int k, T *A, unsigned char *B, T_REDUCE *absmax, T_REDUCE *datatype, T *out, int lda, int ldb, int ldc, int blocksize
) {
    int num_blocks = CDIV(m, 4);
    gemv_4bit_inference_kernel<T, 128, T_REDUCE><<<num_blocks, 128>>>(m, n, k, A, B, absmax, CODE_PARAM, out, lda, ldb, ldc, blocksize);
}

void gemv_4bit_inference_launch_float(
    int m, int n, int k, float *A, unsigned char *B, float *absmax, float *datatype, float *out, int lda, int ldb, int ldc, int blocksize
) {
    int num_blocks = CDIV(m, 4);
    gemv_4bit_inference_kernel_float<float, 128, float>
        <<<num_blocks, 128>>>(m, n, k, A, B, absmax, CODE_PARAM, out, lda, ldb, ldc, blocksize);
}

torch::Tensor gemv_4bit_inference(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor absmax,
    torch::Tensor datatype,
    int blocksize,
    torch::ScalarType dtype,
    std::vector<uint32_t> Bshape
) {
    torch::TensorOptions topts = A.options();
    auto bout = Bshape[0];

    int n = 1;
    int m = Bshape[0];
    int k = Bshape[1];
    int lda = Bshape[0];
    int ldc = Bshape[0];
    int ldb = CDIV(A.sizes()[1], 2);

    torch::Tensor out_;
    if (A.sizes().size() == 3) out_ = torch::empty({A.sizes()[0], A.sizes()[1], bout}, topts);
    else
        out_ = torch::empty({A.sizes()[0], bout}, topts);

    switch (dtype) {
        case torch::kFloat16: {
            TORCH_CHECK(dtype == torch::kFloat16, "Only fp16 dtype is supported for not reduced precision fp16 accumulation")
            TORCH_CHECK(absmax.scalar_type() == torch::kFloat32, "Only fp32 absmax is supported for not reduced precision accumulation")
            TORCH_CHECK(datatype.scalar_type() == torch::kFloat32, "Only fp32 code is supported for not reduced precision accumulation")
            gemv_4bit_inference_launch<nv_half, float>(
                m,
                n,
                k,
                (nv_half *)A.data_ptr(),
                (unsigned char *)B.data_ptr(),
                (float *)absmax.data_ptr(),
                (float *)datatype.data_ptr(),
                (nv_half *)out_.mutable_data_ptr(),
                lda,
                ldb,
                ldc,
                blocksize
            );
            break;
        }
        case torch::kBFloat16: {
            TORCH_CHECK(dtype == torch::kBFloat16, "Only bf16 dtype is supported for not reduced precision accumulation")
            TORCH_CHECK(absmax.scalar_type() == torch::kFloat32, "Only fp32 absmax is supported for not reduced precision accumulation")
            TORCH_CHECK(datatype.scalar_type() == torch::kFloat32, "Only fp32 code is supported for not reduced precision accumulation")
            gemv_4bit_inference_launch<nv_bfloat16, float>(
                m,
                n,
                k,
                (nv_bfloat16 *)A.data_ptr(),
                (unsigned char *)B.data_ptr(),
                (float *)absmax.data_ptr(),
                (float *)datatype.data_ptr(),
                (nv_bfloat16 *)out_.mutable_data_ptr(),
                lda,
                ldb,
                ldc,
                blocksize
            );
            break;
        }
        case torch::kFloat32: {
            TORCH_CHECK(dtype == torch::kFloat32, "Only float32 dtype is supported for float32 gemv_4bit_inference")
            TORCH_CHECK(absmax.scalar_type() == torch::kFloat32, "Only float32 absmax is supported for float32 gemv_4bit_inference")
            TORCH_CHECK(datatype.scalar_type() == torch::kFloat32, "Only float32 code is supported for float32 gemv_4bit_inference")
            gemv_4bit_inference_launch_float(
                m,
                n,
                k,
                (float *)A.data_ptr(),
                (unsigned char *)B.data_ptr(),
                (float *)absmax.data_ptr(),
                (float *)datatype.data_ptr(),
                (float *)out_.mutable_data_ptr(),
                lda,
                ldb,
                ldc,
                blocksize
            );
            break;
        }
        default:
            throw std::runtime_error("Unsupported datatype");
    }
    CUDA_CHECK_RETURN(cudaGetLastError());

    return out_;
}