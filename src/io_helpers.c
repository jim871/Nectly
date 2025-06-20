// src/io_helpers.c

#include "model.h"
#include "parser.h"   // per la dichiarazione di set_decode_params
#include <stdio.h>
#include <stdlib.h>

// ------------------ save_model / load_model ------------------

int save_model(const char *path) {
    printf("save_model: saving model to %s (stub)\n", path);
    return 0;
}

int load_model(const char *path) {
    printf("load_model: loading model from %s (stub)\n", path);
    return 0;
}

// ------------------ set_decode_params ------------------

static int   g_max_len     = 0;
static int   g_top_k       = 0;
static float g_top_p       = 0.0f;
static float g_temperature = 1.0f;

void set_decode_params(int max_len, int top_k, float top_p, float temperature) {
    g_max_len     = max_len;
    g_top_k       = top_k;
    g_top_p       = top_p;
    g_temperature = temperature;
    printf(
      "Decoding params: max_len=%d, top_k=%d, top_p=%.2f, temp=%.2f\n",
      g_max_len, g_top_k, g_top_p, g_temperature
    );
}
