#pragma once

// Dimensione massima del vocabolario (adatta la tua rete)
#define VOCAB_SIZE 50257
#define MAX_TOKEN_LEN 128

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Carica il vocabolario BPE da file.
 * @param path  Percorso al file di vocabolario (es. merges.txt + vocab.json)
 * @return 0 se OK, 1 se errore
 */
int load_bpe_vocab(const char *path);

/**
 * Tokenizza BPE: da stringa a sequenza di token.
 * @param text   Input testuale
 * @param out    Array di output (interi)
 * @param max    Lunghezza massima
 * @return numero di token scritti
 */
int tokenize_bpe(const char *text, int *out, int max);

/**
 * Detokenizza: da sequenza di token a stringa (null-terminated).
 * @param tokens  Array di token
 * @param n       Numero di token
 * @param out     Buffer di output (deve contenere almeno n*MAX_TOKEN_LEN byte)
 */
void detokenize(const int *tokens, int n, char *out);

#ifdef __cplusplus
}
#endif

