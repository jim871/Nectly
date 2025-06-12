#include "model.h"
#include "gpu_helpers.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int *layer_sizes = NULL;
static float **weights = NULL, **biases = NULL;
static size_t n_layers = 0, capacity = 0, input_dim = 0;

int init_model(const char *name) {
    printf("Model '%s' initialized.\n", name);
    // reset dynamic arrays
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
    input_dim = dim;
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

int train_model(const char *path, int epochs, float lr) {
    printf("Training on %s for %d epochs, lr=%f\n", path, epochs, lr);
    int inD = layer_sizes[0];
    float *input = malloc(sizeof(float) * inD);
    float target;
    for (int e = 0; e < epochs; e++) {
        FILE *f = fopen(path, "r");
        if (!f) { perror("Open dataset"); return 1; }
        int count = 0;
        while (fscanf(f, "%f", &input[0]) == 1) {
            for (int i = 1; i < inD; i++) fscanf(f, "%f", &input[i]);
            fscanf(f, "%f", &target);
            // forward pass (GPU)
            float *act = input;
            float *next;
            for (size_t l = 0; l < n_layers; l++) {
                int outD = layer_sizes[l+1];
                next = malloc(sizeof(float) * outD);
                // GPU forward
                float *d_in  = gpu_malloc(sizeof(float) * layer_sizes[l]);
                float *d_w   = gpu_malloc(sizeof(float) * layer_sizes[l] * outD);
                float *d_out = gpu_malloc(sizeof(float) * outD);
                gpu_memcpy_h2d(d_in,  act,                      sizeof(float) * layer_sizes[l]);
                gpu_memcpy_h2d(d_w,   weights[l],              sizeof(float) * layer_sizes[l] * outD);
                launch_matmul(d_in, d_w, d_out, 1, outD, layer_sizes[l]);
                gpu_memcpy_d2h(next, d_out,                    sizeof(float) * outD);
                gpu_free(d_in); gpu_free(d_w); gpu_free(d_out);
                if (l > 0) free(act);
                act = next;
            }
            float delta = act[0] - target;
            // backward pass (CPU, SGD)
            for (int l = n_layers-1; l >= 0; l--) {
                int inD2 = layer_sizes[l], outD2 = layer_sizes[l+1];
                float *prev = (l == 0 ? input : NULL);
                if (l > 0) {
                    prev = malloc(sizeof(float) * layer_sizes[l]);
                    memcpy(prev, input, sizeof(float) * layer_sizes[0]); // simple
                }
                for (int j = 0; j < outD2; j++) {
                    biases[l][j] -= lr * delta;
                    for (int k = 0; k < inD2; k++) {
                        weights[l][j*inD2 + k] -= lr * delta * (l == 0 ? input[k] : prev[k]);
                    }
                }
                if (l > 0) free(prev);
            }
            free(act);
            count++;
        }
        fclose(f);
        printf("Epoch %d done, samples: %d\n", e+1, count);
    }
    free(input);
    return 0;
}

int predict_model(const char *path) {
    printf("Predicting on %s\n", path);
    int inD = layer_sizes[0];
    float *input = malloc(sizeof(float) * inD);
    FILE *f = fopen(path, "r");
    if (!f) { perror("Open dataset"); return 1; }
    while (fscanf(f, "%f", &input[0]) == 1) {
        for (int i = 1; i < inD; i++) fscanf(f, "%f", &input[i]);
        float *act = input;
        float *next;
        for (size_t l = 0; l < n_layers; l++) {
            int outD = layer_sizes[l+1];
            next = malloc(sizeof(float) * outD);
            // GPU forward
            float *d_in  = gpu_malloc(sizeof(float) * layer_sizes[l]);
            float *d_w   = gpu_malloc(sizeof(float) * layer_sizes[l] * outD);
            float *d_out = gpu_malloc(sizeof(float) * outD);
            gpu_memcpy_h2d(d_in,  act,                      sizeof(float) * layer_sizes[l]);
            gpu_memcpy_h2d(d_w,   weights[l],              sizeof(float) * layer_sizes[l] * outD);
            launch_matmul(d_in, d_w, d_out, 1, outD, layer_sizes[l]);
            gpu_memcpy_d2h(next, d_out,                    sizeof(float) * outD);
            gpu_free(d_in); gpu_free(d_w); gpu_free(d_out);
            if (l > 0) free(act);
            act = next;
        }
        printf("Prediction: %f\n", act[0]);
        free(act);
    }
    fclose(f);
    free(input);
    return 0;
}

int save_model(const char *path) {
    FILE *f = fopen(path, "wb");
    fwrite(&n_layers, sizeof(size_t), 1, f);
    fwrite(layer_sizes, sizeof(int), n_layers+1, f);
    for (size_t l = 0; l < n_layers; l++) {
        int inD = layer_sizes[l], outD = layer_sizes[l+1];
        fwrite(weights[l], sizeof(float), inD*outD, f);
        fwrite(biases[l],  sizeof(float), outD,     f);
    }
    fclose(f);
    printf("Model saved to %s\n", path);
    return 0;
}

int load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    fread(&n_layers, sizeof(size_t), 1, f);
    layer_sizes = realloc(layer_sizes, (n_layers+1)*sizeof(int));
    fread(layer_sizes, sizeof(int), n_layers+1, f);
    capacity = n_layers+1;
    weights  = realloc(weights, capacity*sizeof(float*));
    biases   = realloc(biases,  capacity*sizeof(float*));
    for (size_t l = 0; l < n_layers; l++) {
        int inD = layer_sizes[l], outD = layer_sizes[l+1];
        weights[l] = malloc(sizeof(float)*inD*outD);
        biases[l]  = malloc(sizeof(float)*outD);
        fread(weights[l], sizeof(float), inD*outD, f);
        fread(biases[l],  sizeof(float), outD,     f);
    }
    fclose(f);
    printf("Model loaded from %s\n", path);
    return 0;
}
