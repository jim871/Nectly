#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Strutture interne per BPE (da caricare)
static char **vocab = NULL;
static int    vocab_size = 0;

// Stub semplificato: split su spazio
int load_bpe_vocab(const char *path) {
    // Caricamento reale da JSON+merges andrebbe implementato qui.
    // Per ora, riempiamo un vocabolario minimo di prova.
    vocab_size = VOCAB_SIZE;
    vocab = malloc(sizeof(char*) * vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        vocab[i] = NULL;
    }
    printf("Loaded dummy BPE vocab (size %d)\n", vocab_size);
    return 0;
}

int tokenize_bpe(const char *text, int *out, int max) {
    int count = 0;
    const char *p = text;
    while (*p && count < max) {
        // split on space
        const char *q = strchr(p, ' ');
        int len = q ? (q-p) : strlen(p);
        // hash trivial -> token id = len % VOCAB_SIZE
        int id = len % VOCAB_SIZE;
        out[count++] = id;
        if (!q) break;
        p = q + 1;
    }
    return count;
}

void detokenize(const int *tokens, int n, char *out) {
    char *p = out;
    for (int i = 0; i < n; i++) {
        int id = tokens[i];
        // reverse stub -> print id
        p += sprintf(p, "<%d>", id);
    }
    *p = '\0';
}

