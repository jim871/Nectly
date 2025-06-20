#include "parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int nGPUs = 0;
cudaStream_t *streams = NULL;

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <script.nect> [#GPUs]\n", argv[0]);
        return 1;
    }

    // numero di GPU da CLI (opzionale)
    if (argc >= 3) {
        nGPUs = atoi(argv[2]);
    }

    // rileva tutte le GPU disponibili
    int totalGPUs;
    cudaError_t err = cudaGetDeviceCount(&totalGPUs);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    if (nGPUs <= 0 || nGPUs > totalGPUs) {
        nGPUs = totalGPUs;
    }
    printf("Detected %d CUDA device(s); using %d\n", totalGPUs, nGPUs);

    // crea uno stream per cada GPU
    streams = malloc(sizeof(cudaStream_t) * nGPUs);
    if (!streams) {
        fprintf(stderr, "Failed to alloc streams array\n");
        return 1;
    }
    for (int i = 0; i < nGPUs; ++i) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
    }

    // esegui lo script NECT
    int ret = run_script(argv[1], nGPUs, streams);

    // pulizia
    for (int i = 0; i < nGPUs; ++i) {
        cudaSetDevice(i);
        cudaStreamDestroy(streams[i]);
    }
    free(streams);

    return ret;
}



