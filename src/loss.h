#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Calcola la cross-entropy su sequenze:
 *   targets: [seq_len] (token id)
 *   logits:  [seq_len][vocab_size] (pre-softmax)
 *   grads:   [seq_len*vocab_size] (out grad)
 * @return loss media su seq_len
 */
float cross_entropy_loss(const int   *targets,
                         float      **logits,
                         int          seq_len,
                         int          vocab_size,
                         float       *grads);

/**
 * Esegue backprop su tutti i layer:
 *   - utilizza grads da cross_entropy
 *   - aggiorna pesi/bias con AdamW o SGD
 */
void backprop_and_update(float      **activations,
                         float      **weights,
                         float      **biases,
                         int         *layer_sizes,
                         size_t       n_layers,
                         const float *ce_grads,    // seq_len*vocab_size
                         int          seq_len,
                         int          vocab_size,
                         float        lr,
                         float        beta1,
                         float        beta2,
                         float        eps,
                         float        weight_decay,
                         int         *t_step);

#ifdef __cplusplus
}
#endif

