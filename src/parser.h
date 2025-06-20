#pragma once

#include <cuda_runtime.h>

/** Parametri di decoding: lunghezza massima, top-k, top-p, temperatura */
void set_decode_params(int max_len, int top_k, float top_p, float temperature);

/** Altri comandi… */
int run_script(const char *file, int nGPUs, cudaStream_t *streams);
