#include "util.h"
#include <stdlib.h>
#include <math.h>

void random_init(float *arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

float gelu(float x) {
    // Approximated GELU
    return 0.5f * x * (1.0f + tanhf(0.79788456f * x * (1.0f + 0.044715f * x * x)));
}

void layer_norm(float *x, size_t L, float eps) {
    float mean = 0, var = 0;
    for (size_t i = 0; i < L; i++) mean += x[i];
    mean /= L;
    for (size_t i = 0; i < L; i++) {
        float d = x[i] - mean;
        var += d*d;
    }
    var = var / L + eps;
    float inv = 1.0f / sqrtf(var);
    for (size_t i = 0; i < L; i++) {
        x[i] = (x[i] - mean) * inv;
    }
}
