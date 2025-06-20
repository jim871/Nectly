# Makefile per NECT su Windows (NVCC + GNU make)

# Usa nvcc per compilare sia .c che .cu
CC      := nvcc
CFLAGS  := -O2 -std=c99 -I.
LDFLAGS := -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64" -lcudart

# Tutti i sorgenti
SRCS := \
    main.c \
    parser.c \
    model.c \
    tokenizer.c \
    mha.c \
    loss.c \
    optimizer.c \
    gpu_helpers.c \
    util.c \
    kernels.cu

# Oggetti generati
OBJS := $(SRCS:.c=.o)
OBJS := $(OBJS:.cu=.o)

# Nome dell’eseguibile
TARGET := nect.exe

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

# Regola per .c -> .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Regola per .cu -> .o
%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	del /Q $(OBJS) $(TARGET) 2>nul || rm -f $(OBJS) $(TARGET)

