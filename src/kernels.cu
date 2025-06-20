#include <cuda_runtime.h>

// FP32 matrix‐multiply kernel
extern "C" __global__
void matmul_kernel_fp32(const float* A, const float* B, float* C,
                        int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row*K + k] * B[k*N + col];
        }
        C[row*N + col] = sum;
    }
}

// C entry‐point for FP32 matmul
extern "C" void matmul_kernel(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    dim3 block(16,16);
    dim3 grid((N + 15)/16, (M + 15)/16);
    matmul_kernel_fp32<<<grid,block>>>(A, B, C, M, N, K);
}

// FP32 batched matmul kernel
extern "C" __global__
void matmul_batched_kernel_fp32(const float* A, const float* B, float* C,
                                int batch, int M, int N, int K) {
    int b   = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch && row < M && col < N) {
        const float* A_ = A + b*M*K;
        const float* B_ = B + b*K*N;
        float*       C_ = C + b*M*N;
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A_[row*K + k] * B_[k*N + col];
        }
        C_[row*N + col] = sum;
    }
}

// C entry‐point for batched matmul
extern "C" void matmul_batched_kernel(const float* A, const float* B, float* C,
                                      int batch, int M, int N, int K) {
    dim3 block(16,16,1);
    dim3 grid((N + 15)/16, (M + 15)/16, batch);
    matmul_batched_kernel_fp32<<<grid,block>>>(A, B, C, batch, M, N, K);
}
