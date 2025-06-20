// optimizer.h
#pragma once

#include <stddef.h>

void optimizer_step(
    float   **activations,
    float   **weights,
    float   **biases,
    int      *layer_sizes,
    size_t    n_layers,
    const float *grads,    // seq_len * finalD
    int       seq_len,
    int       finalD,
    float     lr,
    float     beta1,
    float     beta2,
    float     eps,
    float     wd,
    int      *t_step
);



