#ifndef PARSER_H
#define PARSER_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Runs a NECT script: parsing, training, and inference.
 * @param script_path Path to the .nect script file.
 * @param nGPUs Number of CUDA devices available.
 * @param streams Array of CUDA streams, one per device.
 * @return 0 on success, non-zero on error.
 */
int run_script(const char *script_path, int nGPUs, cudaStream_t *streams);

#ifdef __cplusplus
}
#endif

#endif // PARSER_H

