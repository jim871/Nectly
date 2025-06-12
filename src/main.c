#include "parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Number of GPUs and streams shared with training code
int nGPUs = 0;
cudaStream_t *streams = NULL;

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <script.nect>\n", argv[0]);
        return 1;
    }

    // Detect available CUDA devices
    cudaError_t err = cudaGetDeviceCount(&nGPUs);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    if (nGPUs < 1) {
        fprintf(stderr, "No CUDA devices found. Exiting.\n");
        return 1;
    }
    printf("Detected %d CUDA device(s)\n", nGPUs);

    // Allocate and create streams for each GPU
    streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nGPUs);
    if (!streams) {
        fprintf(stderr, "Failed to allocate CUDA streams array.\n");
        return 1;
    }
    for (int i = 0; i < nGPUs; ++i) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
    }

    // Execute NECT script (training + inference)
    int ret = run_script(argv[1], nGPUs, streams);

    // Cleanup: destroy streams
    for (int i = 0; i < nGPUs; ++i) {
        cudaSetDevice(i);
        cudaStreamDestroy(streams[i]);
    }
    free(streams);

    return ret;
}
