#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) sum += A[row*K + i] * B[i*N + col];
        C[row*N + col] = sum;
    }
}

extern "C" void launch_matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 threads(16,16);
    dim3 blocks((N+threads.x-1)/threads.x, (M+threads.y-1)/threads.y);
    matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
