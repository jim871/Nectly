#pragma once
#include <stddef.h>
#include <cuda_runtime.h>

// DICHIARAZIONI SOLAMENTE
void* gpu_malloc(size_t size);
void  gpu_free(void *ptr);
void  gpu_memcpy_h2d(void *dst, const void *src, size_t size);
void  gpu_memcpy_d2h(void *dst, const void *src, size_t size);

// Versioni ASINCRONE con stream
void  gpu_memcpy_h2d_async(void *dst, const void *src, size_t size, cudaStream_t stream);
void  gpu_memcpy_d2h_async(void *dst, const void *src, size_t size, cudaStream_t stream);
void  launch_matmul(float *d_A, float *d_B, float *d_C, int M, int N, int K);
void  launch_matmul_async(float *d_A, float *d_B, float *d_C, int M, int N, int K, cudaStream_t stream);

