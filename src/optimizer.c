#include "optimizer.h"
#include <math.h>

void optimizer_step(
    float **weights,
    float **grads,
    float **moments,
    int     *layer_sizes,
    size_t   n_layers,
    const float *ce_grads,
    int      seq_len,
    int      vocab_size,
    float    lr,
    float    beta1,
    float    beta2,
    float    eps,
    float    wd,
    int     *t_step
) {
    // Implementazione di AdamW o SGD a seconda di un flag
    // Esempio: semplice SGD su ce_grads
    for (size_t l = 0; l < n_layers; l++) {
        int inD = layer_sizes[l], outD = layer_sizes[l+1];
        int sz = inD * outD;
        for (int i = 0; i < sz; i++) {
            weights[l][i] -= lr * ce_grads[i];
        }
        // poi aggiorni t_step, moments, ecc. se serve
    }
}





