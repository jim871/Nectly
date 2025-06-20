#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/** Parametri di decoding: lunghezza massima, top-k, top-p, temperatura */
void set_decode_params(int max_len, int top_k, float top_p, float temperature);

/** Campiona il token successivo da un vettore di logits */
int sample_next_token(const float *logits, int vocab_size);

#ifdef __cplusplus
}
#endif
