CC = cl
NVCC = nvcc
CFLAGS = /O2 /I"src" /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include"
LDFLAGS = /link /LIBPATH:"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64" cudart.lib

SRC = src\main.c src\parser.c src\model.c src\util.c src\gpu_helpers.c
OBJ = main.obj parser.obj model.obj util.obj gpu_helpers.obj kernels.obj

all: nect.exe

kernels.obj: kernels.cu
	nvcc -c kernels.cu -o kernels.obj

main.obj: src\main.c
	$(CC) $(CFLAGS) /c src\main.c /Fo$@

parser.obj: src\parser.c
	$(CC) $(CFLAGS) /c src\parser.c /Fo$@

model.obj: src\model.c
	$(CC) $(CFLAGS) /c src\model.c /Fo$@

util.obj: src\util.c
	$(CC) $(CFLAGS) /c src\util.c /Fo$@

gpu_helpers.obj: src\gpu_helpers.c
	$(CC) $(CFLAGS) /c src\gpu_helpers.c /Fo$@

nect.exe: $(OBJ)
	$(CC) $(OBJ) $(LDFLAGS)

clean:
	del /Q *.obj *.exe kernels.obj nect.exe 2>nul || exit 0


