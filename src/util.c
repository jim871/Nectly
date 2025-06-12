#include "util.h"
#include <stdlib.h>
#include <time.h>

void random_init(float *arr, int n) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
}
