#include "parser.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <script.nect>\n", argv[0]);
        return 1;
    }
    return run_script(argv[1]);
}
