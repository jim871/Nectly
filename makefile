CC = gcc
NVCC = nvcc
CFLAGS = -O2 -std=c11 -pthread -I./src
LDFLAGS = -lcudart

SRC = src/main.c src/parser.c src/model.c src/util.c src/gpu_helpers.c
OBJ = $(SRC:.c=.o) kernels.o

all: nect

kernels.o: kernels.cu
	$(NVCC) -c kernels.cu -o kernels.o

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

nect: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f src/*.o kernels.o nect
