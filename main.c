#include "parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Numero di GPU e array di stream condivisi
int nGPUs = 0;
cudaStream_t *streams = NULL;

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <script.nectly>\n", argv[0]);
        return 1;
    }

    // Rileva dispositivi CUDA
    cudaError_t err = cudaGetDeviceCount(&nGPUs);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaGetDeviceCount failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    if (nGPUs < 1) {
        fprintf(stderr, "No CUDA devices found. Exiting.\n");
        return 1;
    }
    printf("Detected %d CUDA device(s)\n", nGPUs);

    // Alloca e crea uno stream per GPU
    streams = malloc(sizeof(cudaStream_t) * nGPUs);
    for (int i = 0; i < nGPUs; ++i) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
    }

    // Esegue lo script NECTLY
    int ret = run_script(argv[1], nGPUs, streams);

    // Cleanup
    for (int i = 0; i < nGPUs; ++i) {
        cudaSetDevice(i);
        cudaStreamDestroy(streams[i]);
    }
    free(streams);
    return ret;
}
