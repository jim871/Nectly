#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

extern "C" void launch_matmul(float *d_A, float *d_B, float *d_C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
}
extern "C" void launch_matmul_async(float *d_A, float *d_B, float *d_C, int M, int N, int K, cudaStream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<blocks, threads, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}
