#pragma once

#include <cuda_runtime.h>
#include "tokenizer.h"

#define MAX_TOKENS    1024
#define EMBEDDING_DIM  64

extern float embedding_matrix[MAX_TOKENS][EMBEDDING_DIM];
extern float embedded_input[MAX_TOKENS][EMBEDDING_DIM];

void init_embedding_layer(void);
void embed_tokens(int *tokens, int count);
void set_fp16(int enabled);
void set_optimizer(int type);

/** Parametri di decoding: lunghezza massima, top-k, top-p, temperatura */
void set_decode_params(int max_len, int top_k, float top_p, float temperature);

int init_model(const char *name);
int set_input_dim(int dim);
int add_layer(int units);
int train_model(const char *path, int epochs, float lr, int nGPUs, cudaStream_t *streams);
int predict_model(const char *path, int nGPUs, cudaStream_t *streams);
void generate_sequence(const char *prompt, int maxlen, int nGPUs, cudaStream_t *streams);
int save_model(const char *path);
int load_model(const char *path);

