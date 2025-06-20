// src/parser.c

#include "parser.h"
#include "model.h"
#include "tokenizer.h"
#include "decode.h"
#include "context.h"    // stub vuoto
#include "server.h"     // stub vuoto
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

int run_script(const char *file, int nGPUs, cudaStream_t *streams) {
    FILE *f = fopen(file, "r");
    if (!f) {
        perror("Open script");
        return 1;
    }

    char line[256], cmd[32];
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "%31s", cmd) != 1) continue;

        if (strcmp(cmd, "model") == 0) {
            char name[64];
            if (sscanf(line, "model %63s", name) == 1) {
                init_model(name);
            }
        }
        else if (strcmp(cmd, "input") == 0) {
            int dim;
            if (sscanf(line, "input %d", &dim) == 1) {
                set_input_dim(dim);
            }
        }
        else if (strcmp(cmd, "layer") == 0) {
            int units;
            if (sscanf(line, "layer %d", &units) == 1) {
                add_layer(units);
            }
        }
        else if (strcmp(cmd, "train") == 0) {
            char path[128];
            int epochs;
            float lr;
            if (sscanf(line, "train %127s %d %f", path, &epochs, &lr) == 3) {
                train_model(path, epochs, lr, nGPUs, streams);
            } else {
                fprintf(stderr, "Error parsing train: %s", line);
            }
        }
        else if (strcmp(cmd, "predict") == 0) {
            char path[128];
            if (sscanf(line, "predict %127s", path) == 1) {
                predict_model(path, nGPUs, streams);
            }
        }
        else if (strcmp(cmd, "save") == 0) {
            char path[128];
            if (sscanf(line, "save %127s", path) == 1) {
                save_model(path);
            }
        }
        else if (strcmp(cmd, "load") == 0) {
            char path[128];
            if (sscanf(line, "load %127s", path) == 1) {
                load_model(path);
            }
        }
        else if (strcmp(cmd, "decode") == 0) {
            int max_len, top_k;
            float top_p, temperature;
            // Esempio di riga: "decode 50 40 0.9 1.2"
            if (sscanf(line, "decode %d %d %f %f",
                       &max_len, &top_k, &top_p, &temperature) == 4) {
                set_decode_params(max_len, top_k, top_p, temperature);
            } else {
                fprintf(stderr, "Error parsing decode: %s", line);
            }
        }
    }

    fclose(f);
    return 0;
}


