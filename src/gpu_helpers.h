// gpu_helpers.h
#pragma once
#include <stddef.h>
#include <cuda_runtime.h>

// Alloc/free e memcpy su GPU
void* gpu_malloc(size_t size);
void  gpu_free(void *ptr);
void  gpu_memcpy_h2d(void *dst, const void *src, size_t size);
void  gpu_memcpy_d2h(void *dst, const void *src, size_t size);

// Lancio del matmul: MxK * KxN = MxN.
// fp16 = 0 → FP32, =1 → FP16 (se supportato dai tuoi kernel)
void launch_matmul(
    const float *d_A,
    const float *d_B,
    float       *d_C,
    int           M,
    int           N,
    int           K
);

// Batched matmul: batch×(MxK) * batch×(KxN) → batch×(MxN)
void launch_matmul_batched(
    const float *d_A,
    const float *d_B,
    float       *d_C,
    int           batch,
    int           M,
    int           N,
    int           K
);


