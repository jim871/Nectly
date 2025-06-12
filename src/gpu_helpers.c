#include "gpu_helpers.h"
#include <cuda_runtime.h>
#include <stdio.h>

void *gpu_malloc(size_t size) {
    void *ptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

void gpu_free(void *ptr) {
    cudaFree(ptr);
}

void gpu_memcpy_h2d(void *d, const void *h, size_t size) {
    cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);
}

void gpu_memcpy_d2h(void *h, const void *d, size_t size) {
    cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);
}

void set_device(int id) {
    int count;
    cudaGetDeviceCount(&count);
    if (id >= 0 && id < count) cudaSetDevice(id);
    else printf("GPU %d not available\n", id);
}
