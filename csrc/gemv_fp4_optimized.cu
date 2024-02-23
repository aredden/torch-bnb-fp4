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
void CUDA_CHECK_RETURN(cudaError_t cudaStatus) {
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Failure: %s\n", cudaGetErrorString(cudaStatus));
        exit(0); // so many segfaults before being able to print out actual crap because of this stupidity
    }
}

template <typename T, int THREADS>
__global__ void gemv_4bit_inference_naive_reduced(
    int M,
    int N,
    int K,
    T *__restrict__ const A,
    unsigned char *B,
    T *absmax,
    const T *datatype,
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
    typedef cub::WarpReduce<T> WarpReduce;
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

    for (int i = threadIdx.x; i < 16; i++)
        quant_map[i] = T(datatype[i]);
    __syncwarp();

    // A: [1, K]
    // B: [N, K]
    for (int inner_idx = warp_lane * num_values_4bit; inner_idx < K; inner_idx += 32 * num_values_4bit) {
        int inner_idx_halved = inner_idx / 2;
        int offset_B = ldb * row_B;
        int absidx = ((2 * offset_B) + inner_idx) / blocksize;
        local_absmax = __ldg(&(absmax[absidx]));

        if (row_B < M) {
            if ((inner_idx_halved + num_values_8bit) < (K / 2)) {
                // this is the most important for performance considerations
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
                    reinterpret_cast<int4 *>(A)[inner_idx / (num_values_4bit / 4) + i];
            } else {
#pragma unroll
                for (int k = 0; k < num_values_4bit / 4; k++)
                    if (inner_idx + (i * num_values_4bit / 4) + k < K)
                        local_A[k] = A[inner_idx + k + (i * num_values_4bit / 4)];
                    else
                        local_A[k] = T(0.0f);
            }

            // accumulate in float; small performance hit for Ampere, but lower error for outputs
#pragma unroll
            for (int k = 0; k < num_values_4bit / 4; k++) {
                local_C += local_A[k] * local_B[k];
                // bf16 multipliation not supported
            }
        }
    }

    local_C = WarpReduce(temp_storage[warp_idx]).Sum(local_C);

    if (row_B < M && warp_lane == 0) out[row_B] = T(local_C);
}

template <typename T, int THREADS>
__global__ void gemv_4bit_inference_naive(
    int M,
    int N,
    int K,
    T *__restrict__ const A,
    unsigned char *B,
    float *absmax,
    const float *datatype,
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
    typedef cub::WarpReduce<float> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[THREADS / 32];

    const int warp_idx = threadIdx.x / 32;
    const int warp_lane = threadIdx.x % 32;
    const int row_B = (THREADS / 32) * blockIdx.x + warp_idx;
    const int num_values_8bit = num_values_4bit / 2;
    float local_C = 0.0f;

    unsigned char local_B_4bit[num_values_8bit];
    T local_B[num_values_4bit / 4];
    T local_A[num_values_4bit / 4];
    __shared__ T quant_map[16];
    T local_absmax = T(0.0f);

    for (int i = threadIdx.x; i < 16; i++)
        quant_map[i] = T(datatype[i]);
    __syncwarp();
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
                    reinterpret_cast<int4 *>(A)[inner_idx / (num_values_4bit / 4) + i];
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
                local_C += (float)(local_A[k] * local_B[k]);
            }
        }
    }

    local_C = WarpReduce(temp_storage[warp_idx]).Sum(local_C);

    if (row_B < M && warp_lane == 0) {
        out[row_B] = T(local_C);
    };
}
void gemv_4bit_inference_naive_fp16_reduc(
    int m,
    int n,
    int k,
    half *A,
    unsigned char *B,
    half *absmax,
    half *datatype,
    half *out,
    int lda,
    int ldb,
    int ldc,
    int blocksize
) {
    int num_blocks = (m + 3) / 4;
    gemv_4bit_inference_naive_reduced<half, 128>
        <<<num_blocks, 128>>>(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize);
}
void gemv_4bit_inference_naive_bf16_reduc(
    int m,
    int n,
    int k,
    nv_bfloat16 *A,
    unsigned char *B,
    nv_bfloat16 *absmax,
    nv_bfloat16 *datatype,
    nv_bfloat16 *out,
    int lda,
    int ldb,
    int ldc,
    int blocksize
) {
    int num_blocks = (m + 3) / 4;
    gemv_4bit_inference_naive_reduced<nv_bfloat16, 128>
        <<<num_blocks, 128>>>(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize);
}
void gemv_4bit_inference_naive_fp16_reduc(
    int m,
    int n,
    int k,
    float *A,
    unsigned char *B,
    float *absmax,
    float *datatype,
    float *out,
    int lda,
    int ldb,
    int ldc,
    int blocksize
) {
    int num_blocks = (m + 3) / 4;
    gemv_4bit_inference_naive_reduced<float, 128>
        <<<num_blocks, 128>>>(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize);
}

void gemv_4bit_inference_naive_fp16(
    int m,
    int n,
    int k,
    half *A,
    unsigned char *B,
    float *absmax,
    float *datatype,
    half *out,
    int lda,
    int ldb,
    int ldc,
    int blocksize
) {
    int num_blocks = (m + 3) / 4;
    gemv_4bit_inference_naive<half, 128>
        <<<num_blocks, 128>>>(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize);
}
void gemv_4bit_inference_naive_bf16(
    int m,
    int n,
    int k,
    nv_bfloat16 *A,
    unsigned char *B,
    float *absmax,
    float *datatype,
    nv_bfloat16 *out,
    int lda,
    int ldb,
    int ldc,
    int blocksize
) {
    int num_blocks = (m + 3) / 4;
    gemv_4bit_inference_naive<nv_bfloat16, 128>
        <<<num_blocks, 128>>>(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize);
}
void gemv_4bit_inference_naive_fp32(
    int m,
    int n,
    int k,
    float *A,
    unsigned char *B,
    float *absmax,
    float *datatype,
    float *out,
    int lda,
    int ldb,
    int ldc,
    int blocksize
) {
    int num_blocks = (m + 3) / 4;
    gemv_4bit_inference_naive<float, 128>
        <<<num_blocks, 128>>>(m, n, k, A, B, absmax, datatype, out, lda, ldb, ldc, blocksize);
}

torch::Tensor gemv_4bit_inference(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor absmax,
    torch::Tensor datatype,
    int blocksize,
    torch::ScalarType dtype,
    std::vector<uint32_t> Bshape,
    bool use_reduced_prec_accumulate
) {

    auto topts = torch::TensorOptions().dtype(dtype).device(A.device());
    auto bout = Bshape[0];

    int n = 1;
    int m = Bshape[0];
    int k = Bshape[1];
    int lda = Bshape[0];
    int ldc = Bshape[0];
    int ldb = (A.sizes()[1] + 1) / 2;

    torch::Tensor out_;
    if (A.sizes().size() == 3) out_ = torch::empty({A.sizes()[0], A.sizes()[1], bout}, topts);
    else
        out_ = torch::empty({A.sizes()[0], bout}, topts);

    switch (dtype) {
        case torch::kFloat16:
            if (use_reduced_prec_accumulate) {
                TORCH_CHECK(dtype == torch::kFloat16, "Only fp16 dtype is supported for reduced precision accumulation")
                TORCH_CHECK(
                    absmax.scalar_type() == torch::kFloat16,
                    "Only fp16 absmax is supported for reduced precision accumulation"
                )
                TORCH_CHECK(
                    datatype.scalar_type() == torch::kFloat16,
                    "Only fp16 code is supported for reduced precision accumulation"
                )
                gemv_4bit_inference_naive_fp16_reduc(
                    m,
                    n,
                    k,
                    (half *)A.data_ptr(),
                    (unsigned char *)B.data_ptr(),
                    (half *)absmax.data_ptr(),
                    (half *)datatype.data_ptr(),
                    (half *)out_.mutable_data_ptr(),
                    lda,
                    ldb,
                    ldc,
                    blocksize
                );
            } else {
                gemv_4bit_inference_naive_fp16(
                    m,
                    n,
                    k,
                    (half *)A.data_ptr(),
                    (unsigned char *)B.data_ptr(),
                    (float *)absmax.data_ptr(),
                    (float *)datatype.data_ptr(),
                    (half *)out_.mutable_data_ptr(),
                    lda,
                    ldb,
                    ldc,
                    blocksize
                );
            }
            break;
        case torch::kBFloat16:
            if (use_reduced_prec_accumulate) {
                TORCH_CHECK(
                    dtype == torch::kBFloat16, "Only bf16 dtype is supported for reduced precision accumulation"
                )
                TORCH_CHECK(
                    absmax.scalar_type() == torch::kBFloat16,
                    "Only bf16 absmax is supported for reduced precision accumulation"
                )
                TORCH_CHECK(
                    datatype.scalar_type() == torch::kBFloat16,
                    "Only bf16 code is supported for reduced precision accumulation"
                )
                gemv_4bit_inference_naive_bf16_reduc(
                    m,
                    n,
                    k,
                    (nv_bfloat16 *)A.data_ptr(),
                    (unsigned char *)B.data_ptr(),
                    (nv_bfloat16 *)absmax.data_ptr(),
                    (nv_bfloat16 *)datatype.data_ptr(),
                    (nv_bfloat16 *)out_.mutable_data_ptr(),
                    lda,
                    ldb,
                    ldc,
                    blocksize
                );
            } else {
                gemv_4bit_inference_naive_bf16(
                    m,
                    n,
                    k,
                    (__nv_bfloat16 *)A.data_ptr(),
                    (unsigned char *)B.data_ptr(),
                    (float *)absmax.data_ptr(),
                    (float *)datatype.data_ptr(),
                    (__nv_bfloat16 *)out_.mutable_data_ptr(),
                    lda,
                    ldb,
                    ldc,
                    blocksize
                );
            }
            break;
        case torch::kFloat32:
            gemv_4bit_inference_naive_fp32(
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
        default:
            throw std::runtime_error("Unsupported datatype");
    }
    CUDA_CHECK_RETURN(cudaGetLastError());

    return out_;
}