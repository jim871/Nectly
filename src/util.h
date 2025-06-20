#pragma once
#include <stddef.h>

/** Inizializza array con numeri random [-1,1]. */
void random_init(float *arr, size_t n);

/** Funzione GELU. */
float gelu(float x);

/** Layer normalization su vettore L. */
void layer_norm(float *x, size_t L, float eps);
