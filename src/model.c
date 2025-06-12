#include "model.h"
#include "gpu_helpers.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>





static int *layer_sizes = NULL;
static float **weights = NULL, **biases = NULL;
static size_t n_layers = 0, capacity = 0;


int save_model(const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) return 1;

    fwrite(&n_layers, sizeof(size_t), 1, f);
    fwrite(layer_sizes, sizeof(int), n_layers + 1, f);

    for (size_t l = 0; l < n_layers; l++) {
        int inD = layer_sizes[l];
        int outD = layer_sizes[l+1];
        fwrite(weights[l], sizeof(float), inD * outD, f);
        fwrite(biases[l], sizeof(float), outD, f);
    }

    fclose(f);
    printf("Model saved to %s\n", filename);
    return 0;
}

int load_model(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return 1;

    fread(&n_layers, sizeof(size_t), 1, f);
    if (capacity < n_layers) {
        capacity = n_layers;
        weights = realloc(weights, capacity * sizeof(float*));
        biases = realloc(biases, capacity * sizeof(float*));
    }

    layer_sizes = realloc(layer_sizes, (n_layers + 1) * sizeof(int));
    fread(layer_sizes, sizeof(int), n_layers + 1, f);

    for (size_t l = 0; l < n_layers; l++) {
        int inD = layer_sizes[l];
        int outD = layer_sizes[l+1];
        weights[l] = malloc(sizeof(float) * inD * outD);
        biases[l]  = malloc(sizeof(float) * outD);
        fread(weights[l], sizeof(float), inD * outD, f);
        fread(biases[l], sizeof(float), outD, f);
    }

    fclose(f);
    printf("Model loaded from %s\n", filename);
    return 0;
}

int init_model(const char *name) {
    printf("Model '%s' initialized.\n", name);
    for (size_t i = 0; i < n_layers; i++) {
        free(weights[i]);
        free(biases[i]);
    }
    free(weights); free(biases); free(layer_sizes);
    capacity = 4;
    layer_sizes = malloc((capacity + 1) * sizeof(int));
    weights = malloc(capacity * sizeof(float*));
    biases  = malloc(capacity * sizeof(float*));
    n_layers = 0;
    layer_sizes[0] = 0;
    return 0;
}

int set_input_dim(int dim) {
    layer_sizes[0] = dim;
    printf("Input dimension set to %d\n", dim);
    return 0;
}

int add_layer(int units) {
    if (n_layers >= capacity) {
        capacity *= 2;
        weights = realloc(weights, capacity * sizeof(float*));
        biases  = realloc(biases,  capacity * sizeof(float*));
        layer_sizes = realloc(layer_sizes, (capacity + 1) * sizeof(int));
    }
    size_t idx = n_layers;
    int inD = layer_sizes[idx];
    layer_sizes[idx+1] = units;
    weights[idx] = malloc(sizeof(float) * inD * units);
    biases[idx]  = malloc(sizeof(float) * units);
    random_init(weights[idx], inD * units);
    memset(biases[idx], 0, sizeof(float) * units);
    n_layers++;
    printf("Added layer %zu: %d units\n", idx, units);
    return 0;
}

int train_model(const char *path, int epochs, float lr, int nGPUs, cudaStream_t *streams) {
    printf("Training on %s for %d epochs, lr=%.8f\n", path, epochs, lr);
    FILE *f = fopen(path, "r");
    if (!f) { perror("open dataset"); return 1; }

    char buffer[256];

    for (int epoch = 0; epoch < epochs; epoch++) {
        rewind(f);
        float x[64], y;
        int sample_count = 0;
        while (fgets(buffer, sizeof(buffer), f)) {
            if (sscanf(buffer, "%f,%f,%f:%f", &x[0], &x[1], &x[2], &y) != 4) continue;
            sample_count++;
            float **activations = malloc((n_layers + 1) * sizeof(float*));
            activations[0] = malloc(sizeof(float) * layer_sizes[0]);
            memcpy(activations[0], x, sizeof(float) * layer_sizes[0]);

            for (size_t l = 0; l < n_layers; l++) {
                int gpu_id = l % nGPUs;
                cudaSetDevice(gpu_id);
                int inD = layer_sizes[l], outD = layer_sizes[l+1];
                float *next = malloc(sizeof(float) * outD);
                float *d_in = gpu_malloc(sizeof(float) * inD);
                float *d_w  = gpu_malloc(sizeof(float) * inD * outD);
                float *d_out = gpu_malloc(sizeof(float) * outD);
                gpu_memcpy_h2d(d_in, activations[l], sizeof(float) * inD);
                gpu_memcpy_h2d(d_w, weights[l], sizeof(float) * inD * outD);
                launch_matmul(d_in, d_w, d_out, 1, outD, inD);
                gpu_memcpy_d2h(next, d_out, sizeof(float) * outD);
                for (int j = 0; j < outD; j++) {
                    next[j] += biases[l][j];
                    next[j] = tanh(next[j]);
                }
                gpu_free(d_in); gpu_free(d_w); gpu_free(d_out);
                activations[l+1] = next;
            }

            float **deltas = malloc(n_layers * sizeof(float*));
            for (int l = n_layers - 1; l >= 0; l--) {
                int inD = layer_sizes[l], outD = layer_sizes[l+1];
                deltas[l] = malloc(sizeof(float) * outD);
                for (int j = 0; j < outD; j++) {
                    float out = activations[l+1][j];
                    float d = 0;
                    if (l == n_layers - 1) {
                        d = 2.0f * (out - y) * (1 - out * out);
                    } else {
                        for (int k = 0; k < layer_sizes[l+2]; k++) {
                            d += deltas[l+1][k] * weights[l+1][j * layer_sizes[l+2] + k];
                        }
                        d *= (1 - out * out);
                    }
                    deltas[l][j] = d;
                }
            }

            for (size_t l = 0; l < n_layers; l++) {
                int inD = layer_sizes[l], outD = layer_sizes[l+1];
                for (int j = 0; j < outD; j++) {
                    for (int i = 0; i < inD; i++) {
                        weights[l][i * outD + j] -= lr * deltas[l][j] * activations[l][i];
                    }
                    biases[l][j] -= lr * deltas[l][j];
                }
            }

            for (size_t l = 0; l <= n_layers; l++) free(activations[l]);
            for (size_t l = 0; l < n_layers; l++) free(deltas[l]);
            free(activations);
            free(deltas);
        }
        printf("Epoch %d done, samples: %d\n", epoch + 1, sample_count);
    }

    fclose(f);
    return 0;
}

int predict_model(const char *path, int nGPUs, cudaStream_t *streams) {
    printf("Predicting on %s\n", path);
    FILE *f = fopen(path, "r");
    if (!f) { perror("open dataset"); return 1; }

    char buffer[256];
    float x[64], y;
    int line = 0;

    while (fgets(buffer, sizeof(buffer), f)) {
        if (sscanf(buffer, "%f,%f,%f:%f", &x[0], &x[1], &x[2], &y) != 4) continue;
        line++;
        printf("[Predict %d] Input: [%f, %f, %f] Target: %f\n", line, x[0], x[1], x[2], y);
        float *act = malloc(sizeof(float) * layer_sizes[0]);
        memcpy(act, x, sizeof(float) * layer_sizes[0]);

        for (size_t l = 0; l < n_layers; l++) {
            int gpu_id = l % nGPUs;
            cudaSetDevice(gpu_id);
            int inD = layer_sizes[l], outD = layer_sizes[l+1];
            float *next = malloc(sizeof(float) * outD);
            float *d_in = gpu_malloc(sizeof(float) * inD);
            float *d_w  = gpu_malloc(sizeof(float) * inD * outD);
            float *d_out = gpu_malloc(sizeof(float) * outD);
            gpu_memcpy_h2d(d_in, act, sizeof(float) * inD);
            gpu_memcpy_h2d(d_w, weights[l], sizeof(float) * inD * outD);
            launch_matmul(d_in, d_w, d_out, 1, outD, inD);
            gpu_memcpy_d2h(next, d_out, sizeof(float) * outD);
            for (int j = 0; j < outD; j++) {
                next[j] += biases[l][j];
                next[j] = tanh(next[j]);
            }
            gpu_free(d_in); gpu_free(d_w); gpu_free(d_out);
            free(act);
            act = next;
        }

        printf("[Predict %d] Output: %f\n", line, act[0]);
        free(act);
    }

    fclose(f);
    return 0;
}
