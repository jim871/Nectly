#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int   input_dim;
    int   num_heads;
    int   head_dim;
    // CPU weights
    float *W_q, *W_k, *W_v, *W_o;
    // GPU weights
    float *d_W_q, *d_W_k, *d_W_v, *d_W_o;
    // CUDA stream (opzionale)
    cudaStream_t stream;
} MHALayer;

/**
 * Alloca e inizializza un layer MHA.
 * @param input_dim dimensione del modell (es. d_model)
 * @param num_heads numero di teste H
 * @param head_dim  dimensione di ciascuna testa D_head
 * @return puntatore a MHALayer inizializzato
 */
MHALayer *init_mha_layer(int input_dim, int num_heads, int head_dim);

/**
 * Forward batched MHA: in [B, seq_len, d_model] out [B, seq_len, d_model].
 * Per semplicità B=1 nel tuo caso, ma puoi estendere la firma.
 */
void forward_mha(const MHALayer *layer,
                 const float *in,   // size = seq_len * input_dim
                 float       *out,  // same size
                 int seq_len);

/** Dealloca le risorse del layer. */
void free_mha_layer(MHALayer *layer);

#ifdef __cplusplus
}
#endif

