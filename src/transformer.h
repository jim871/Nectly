// src/transformer.h
#pragma once

/**
 * Configura il Transformer: numero di layer, heads, hidden dim e FFN dim
 */
void set_transformer_config(int layers, int heads, int hidden, int ffn);


// src/transformer.c
#include "transformer.h"
#include <stdio.h>

void set_transformer_config(int layers, int heads, int hidden, int ffn) {
    printf("Transformer config: layers=%d, heads=%d, hidden=%d, ffn=%d\n", layers, heads, hidden, ffn);
    // Stub: in futuro alloca strutture e inizializza pesi
}
