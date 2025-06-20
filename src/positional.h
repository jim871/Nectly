// src/positional.h
#pragma once

/**
 * Configura il tipo di positional encoding ("sinusoidal" o "learnable").
 */
void set_positional_encoding(const char *type);

/**
 * Applica l'encoding posizionale su un embedding
 * emb: array di dimensione seq_len * dim
 */
void apply_positional(float *emb, int seq_len, int dim);


// src/positional.c
#include "positional.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static char encoding_type[32] = "sinusoidal";

void set_positional_encoding(const char *type) {
    strncpy(encoding_type, type, 31);
    encoding_type[31] = '\0';
    printf("Positional encoding set to %s\n", encoding_type);
}

void apply_positional(float *emb, int seq_len, int dim) {
    // Stub: in futuro implementa sinusoidal o learnable
    if (strcmp(encoding_type, "sinusoidal") == 0) {
        for (int pos = 0; pos < seq_len; pos++) {
            for (int i = 0; i < dim; i += 2) {
                float angle = pos / powf(10000.0f, (float)i / dim);
                emb[pos*dim + i]     += sinf(angle);
                if (i+1 < dim) emb[pos*dim + i+1] += cosf(angle);
            }
        }
    }
    // else: no-op
}