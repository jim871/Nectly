// src/model.c
#include "model.h"
#include "gpu_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>



// Stato del modello
static int   *layer_sizes = NULL;
static float **weights    = NULL;
static float **biases     = NULL;
static size_t n_layers    = 0, capacity = 0;

/** Inizializza modello vuoto */
int init_model(const char *name) {
    printf("Model '%s' initialized.\n", name);
    for (size_t i = 0; i < n_layers; i++) {
        free(weights[i]);
        free(biases[i]);
    }
    free(weights); free(biases); free(layer_sizes);

    capacity = 4;
    layer_sizes = malloc((capacity + 1) * sizeof(int));
    weights     = malloc(capacity * sizeof(float*));
    biases      = malloc(capacity * sizeof(float*));
    n_layers    = 0;
    layer_sizes[0] = 0;
    return 0;
}

/** Imposta dimensione input */
int set_input_dim(int dim) {
    layer_sizes[0] = dim;
    printf("Input dimension set to %d\n", dim);
    return 0;
}

/** Aggiunge un dense layer */
int add_layer(int units) {
    if (n_layers >= capacity) {
        capacity *= 2;
        weights     = realloc(weights,     capacity * sizeof(float*));
        biases      = realloc(biases,      capacity * sizeof(float*));
        layer_sizes = realloc(layer_sizes, (capacity + 1) * sizeof(int));
    }
    int idx = (int)n_layers;
    int inD = layer_sizes[idx];
    layer_sizes[idx+1] = units;
    weights[idx] = malloc(inD * units * sizeof(float));
    biases[idx]  = malloc(units * sizeof(float));
    // inizializzazione uniforme [-1,1]
    for (int i = 0; i < inD * units; i++) {
        weights[idx][i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
    memset(biases[idx], 0, units * sizeof(float));
    n_layers++;
    printf("Added layer %d: %d units\n", idx, units);
    return 0;
}

/** Training MSE + backprop manuale, multi-GPU batch-parallel */
int train_model(const char *path, int epochs, float lr, int nGPUs, cudaStream_t *streams) {
    FILE *f = fopen(path, "r");
    if (!f) { perror("open dataset"); return 1; }

    // buffer per le attivazioni e i delta
    float **a    = malloc((n_layers+1) * sizeof(float*));
    float **dels = malloc(n_layers * sizeof(float*));

    for (int e = 0; e < epochs; e++) {
        rewind(f);
        int count = 0;
        char line[1024];

        while (fgets(line, sizeof(line), f)) {
            // parse CSV x0,x1,...,xN,target
            float *x = malloc(layer_sizes[0] * sizeof(float));
            float target;
            char *tok = strtok(line, ",");
            int i = 0;
            for (; i < layer_sizes[0] && tok; i++) {
                x[i] = atof(tok);
                tok = strtok(NULL, ",");
            }
            if (!tok) { free(x); continue; }
            target = atof(tok);

            // forward
            a[0] = x;
            for (size_t l = 0; l < n_layers; l++) {
                int inD = layer_sizes[l], outD = layer_sizes[l+1];
                a[l+1] = malloc(outD * sizeof(float));
                int gpu = (int)(l % nGPUs);
                cudaSetDevice(gpu);

                // GPU matmul
                float *d_in  = gpu_malloc(inD * sizeof(float));
                float *d_w   = gpu_malloc(inD * outD * sizeof(float));
                float *d_out = gpu_malloc(outD * sizeof(float));

                gpu_memcpy_h2d(d_in, a[l], inD * sizeof(float));
                gpu_memcpy_h2d(d_w, weights[l], inD * outD * sizeof(float));
                launch_matmul(d_in, d_w, d_out, 1, outD, inD);
                gpu_memcpy_d2h(a[l+1], d_out, outD * sizeof(float));

                // bias + tanh
                for (int j = 0; j < outD; j++) {
                    a[l+1][j] = tanh(a[l+1][j] + biases[l][j]);
                }

                gpu_free(d_in);
                gpu_free(d_w);
                gpu_free(d_out);
            }

            // backward (MSE)
            int L = (int)n_layers;
            dels[L-1] = malloc(layer_sizes[L] * sizeof(float));
            float pred = a[L][0];
            float diff = pred - target;
            // gradiente solo sulla prima unità
            for (int j = 0; j < layer_sizes[L]; j++) {
                dels[L-1][j] = (j == 0 ? 2*diff * (1 - pred*pred) : 0.0f);
            }
            // backprop livelli intermedi
            for (int l = L-2; l >= 0; l--) {
                int outD = layer_sizes[l+1], inD = layer_sizes[l];
                dels[l] = malloc(outD * sizeof(float));
                for (int j = 0; j < outD; j++) {
                    float sum = 0;
                    for (int k = 0; k < layer_sizes[l+2]; k++) {
                        sum += dels[l+1][k] * weights[l+1][j*layer_sizes[l+2] + k];
                    }
                    float act = a[l+1][j];
                    dels[l][j] = sum * (1 - act*act);
                }
            }

            // update parametri
            for (size_t l = 0; l < n_layers; l++) {
                int inD = layer_sizes[l], outD = layer_sizes[l+1];
                for (int j = 0; j < outD; j++) {
                    for (int i2 = 0; i2 < inD; i2++) {
                        weights[l][i2*outD + j] -= lr * dels[l][j] * a[l][i2];
                    }
                    biases[l][j] -= lr * dels[l][j];
                }
            }

            // libera memoria temporanea
            for (size_t l = 0; l <= n_layers; l++) {
                if (l > 0) free(dels[l-1]);
                if (l < n_layers) free(a[l+1]);
            }
            free(x);
            count++;
        }
        printf("Epoch %d done: %d samples\n", e+1, count);
    }

    free(a);
    free(dels);
    fclose(f);
    return 0;
}

/** Stub per predict – estendi come vuoi */
int predict_model(const char *path, int nGPUs, cudaStream_t *streams) {
    printf("predict_model: not yet implemented\n");
    return 0;
}
