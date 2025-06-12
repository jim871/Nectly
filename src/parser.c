#include "parser.h"
#include "model.h"
#include <stdio.h>
#include <string.h>

int run_script(const char *file, int nGPUs, cudaStream_t *streams) {
    FILE *f = fopen(file, "r");
    if (!f) { perror("Open script"); return 1; }

    char line[256], cmd[32];

    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "%31s", cmd) != 1) continue;

        if (strcmp(cmd, "model") == 0) {
            char name[64];
            sscanf(line, "model %63s", name);
            init_model(name);

        } else if (strcmp(cmd, "input") == 0) {
            int d;
            sscanf(line, "input %d", &d);
            set_input_dim(d);

        } else if (strcmp(cmd, "layer") == 0) {
            int u;
            sscanf(line, "layer %d", &u);
            add_layer(u);

        } else if (strcmp(cmd, "train") == 0) {
            char p[128];
            int e;
            float lr;
            if (sscanf(line, "train %127s %d %f", p, &e, &lr) == 3) {
                printf("Parsed train command: file=%s epochs=%d lr=%f\n", p, e, lr);
                train_model(p, e, lr, nGPUs, streams);
            } else {
                fprintf(stderr, "Errore parsing comando train: %s", line);
            }

        } else if (strcmp(cmd, "predict") == 0) {
            char p[128];
            sscanf(line, "predict %127s", p);
            predict_model(p, nGPUs, streams);

        } else if (strcmp(cmd, "save") == 0) {
            char p[128];
            sscanf(line, "save %127s", p);
            save_model(p);

        } else if (strcmp(cmd, "load") == 0) {
            char p[128];
            sscanf(line, "load %127s", p);
            load_model(p);
        }
    }

    fclose(f);
    return 0;
}

