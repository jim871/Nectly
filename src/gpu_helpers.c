#include <stdio.h>          // per stderr
#include "gpu_helpers.h"
#include <cuda_runtime.h>

void* gpu_malloc(size_t size) {
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    return ptr;
}

void gpu_free(void *ptr) {
    if (ptr) cudaFree(ptr);
}

void gpu_memcpy_h2d(void *dst, const void *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void gpu_memcpy_d2h(void *dst, const void *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void launch_matmul(
    const float *d_A,
    const float *d_B,
    float       *d_C,
    int           M,
    int           N,
    int           K
) {
    // chiama il kernel custom in kernels.cu
    matmul_kernel(d_A, d_B, d_C, M, N, K);
}

void launch_matmul_batched(
    const float *d_A,
    const float *d_B,
    float       *d_C,
    int           batch,
    int           M,
    int           N,
    int           K
) {
    matmul_batched_kernel(d_A, d_B, d_C, batch, M, N, K);
}
