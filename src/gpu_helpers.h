#pragma once
#include <stddef.h>
void launch_matmul(const float* A, const float* B, float* C, int M, int N, int K);

void *gpu_malloc(size_t size);
void  gpu_free(void *ptr);
void  gpu_memcpy_h2d(void *d, const void *h, size_t size);
void  gpu_memcpy_d2h(void *h, const void *d, size_t size);
void  set_device(int id);
