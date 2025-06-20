#include "mha.h"
#include "gpu_helpers.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Helper CPU softmax
static void softmax(float *x, int len) {
    float mx = x[0];
    for (int i = 1; i < len; i++) mx = fmaxf(mx, x[i]);
    float sum = 0;
    for (int i = 0; i < len; i++) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    for (int i = 0; i < len; i++) x[i] /= sum;
}

MHALayer *init_mha_layer(int input_dim, int num_heads, int head_dim) {
    int E = num_heads * head_dim;
    MHALayer *m = malloc(sizeof(*m));
    m->input_dim = input_dim;
    m->num_heads = num_heads;
    m->head_dim  = head_dim;
    // Alloca W_q,k,v,o su CPU
    m->W_q = malloc(sizeof(float)*input_dim*E);
    m->W_k = malloc(sizeof(float)*input_dim*E);
    m->W_v = malloc(sizeof(float)*input_dim*E);
    m->W_o = malloc(sizeof(float)*E*input_dim);
    random_init(m->W_q, input_dim*E);
    random_init(m->W_k, input_dim*E);
    random_init(m->W_v, input_dim*E);
    random_init(m->W_o, E*input_dim);
    // Copia su GPU
    cudaStreamCreate(&m->stream);
    m->d_W_q = gpu_malloc(sizeof(float)*input_dim*E);
    m->d_W_k = gpu_malloc(sizeof(float)*input_dim*E);
    m->d_W_v = gpu_malloc(sizeof(float)*input_dim*E);
    m->d_W_o = gpu_malloc(sizeof(float)*E*input_dim);
    gpu_memcpy_h2d(m->d_W_q, m->W_q, sizeof(float)*input_dim*E);
    gpu_memcpy_h2d(m->d_W_k, m->W_k, sizeof(float)*input_dim*E);
    gpu_memcpy_h2d(m->d_W_v, m->W_v, sizeof(float)*input_dim*E);
    gpu_memcpy_h2d(m->d_W_o, m->W_o, sizeof(float)*E*input_dim);
    return m;
}

void forward_mha(const MHALayer *layer,
                 const float *in, float *out, int seq_len)
{
    int H = layer->num_heads;
    int D = layer->head_dim;
    int E = H*D;
    // 1) Q = in × W_q, K = in × W_k, V = in × W_v
    float *Q = malloc(sizeof(float)*seq_len*E);
    float *K = malloc(sizeof(float)*seq_len*E);
    float *V = malloc(sizeof(float)*seq_len*E);
    // Usa gpu_helpers launch_matmul
    launch_matmul(in, layer->d_W_q, Q, seq_len, E, layer->input_dim);
    launch_matmul(in, layer->d_W_k, K, seq_len, E, layer->input_dim);
    launch_matmul(in, layer->d_W_v, V, seq_len, E, layer->input_dim);
    // 2) per ogni head h,  calcola attention
    for (int h = 0; h < H; h++) {
        int off = h * D;
        for (int t = 0; t < seq_len; t++) {
            float scores[128];  // presuppone seq_len<=128, per ora OK
            for (int t2 = 0; t2 < seq_len; t2++) {
                float s = 0;
                for (int d = 0; d < D; d++) {
                    s += Q[t*E + off + d] * K[t2*E + off + d];
                }
                scores[t2] = s / sqrtf((float)D);
            }
            softmax(scores, seq_len);
            // output[t][off..off+D] = sum_{t2} scores[t2] * V[t2][off..off+D]
            for (int d = 0; d < D; d++) {
                float v = 0;
                for (int t2 = 0; t2 < seq_len; t2++) {
                    v += scores[t2] * V[t2*E + off + d];
                }
                out[t*E + off + d] = v;
            }
        }
    }
    // 3) Proiezione finale out = out × W_o
    float *tmp = malloc(sizeof(float)*seq_len*layer->input_dim);
    launch_matmul(out, layer->d_W_o, tmp, seq_len, layer->input_dim, E);
    memcpy(out, tmp, sizeof(float)*seq_len*layer->input_dim);
    free(Q); free(K); free(V); free(tmp);
}

void free_mha_layer(MHALayer *layer) {
    free(layer->W_q); free(layer->W_k); free(layer->W_v); free(layer->W_o);
    gpu_free(layer->d_W_q); gpu_free(layer->d_W_k);
    gpu_free(layer->d_W_v); gpu_free(layer->d_W_o);
    cudaStreamDestroy(layer->stream);
    free(layer);
}

