# 🚀 NECTLY – Neural C Training Language with CUDA Support

**NECTLY** è un linguaggio di programmazione standalone scritto in **C puro** con supporto a **CUDA**, progettato per **definire, addestrare e inferire modelli di deep learning** direttamente, senza dipendenze esterne.



## 📦 Requisiti minimi

- **Sistema operativo**: Windows 10/11 (x64)
- **Visual Studio 2022** (con C++ e toolchain MSVC)
- **CUDA Toolkit** 12.0 o superiore
- **GPU NVIDIA** compatibile (Compute Capability ≥ 5.0)
- (Facoltativo) **MSYS2** o **WSL** per usare `make`

---

## ⚙️ Installazione (Windows)

### 1. Installa gli strumenti necessari

- [Visual Studio 2022](https://visualstudio.microsoft.com/) con componenti per C++
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- *(Facoltativo)* [MSYS2](https://www.msys2.org/) per compilare con `make`

---

### 2. Clona il repository

```bash
git clone https://github.com/jim871/Nectly.git
cd Nectly
```

---

### 3. Compilazione manuale (senza make)

Apri **Developer Command Prompt for VS 2022** (x64), poi:

```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
set PATH=%CUDA_PATH%\bin;%PATH%
set LIB=%CUDA_PATH%\lib\x64;%LIB%
set INCLUDE=%CUDA_PATH%\include;%INCLUDE%

cl /c /MD /O2 /I src src\*.c main.c
nvcc -c -O2 -ccbin "cl" -Xcompiler /MD -I src src\kernels.cu -o kernels.obj
cl *.obj /link /OUT:nect.exe /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib
```

---

## 🧪 Esempio di utilizzo

### File `dataset.txt`

```
0.1 0.2 0.3 : 1.0
0.4 0.5 0.6 : 0.0
0.7 0.8 0.9 : 1.0
```

### File `example.nectly`

```
model Demo
input 3
layer 4
layer 1
train dataset.txt epochs 100 lr 0.01
predict dataset.txt
save model.bin
load model.bin
predict dataset.txt
```

### Esecuzione:

```bash
nect.exe example.nectly
```

---

## 📖 Sintassi `.nectly`

| Comando                                | Descrizione                            |
|----------------------------------------|----------------------------------------|
| `model <nome>`                         | Crea un modello                        |
| `input <N>`                            | Imposta la dimensione dell’input       |
| `layer <neuroni>`                      | Aggiunge un layer fully-connected      |
| `train <file> epochs <N> lr <f>`       | Addestra il modello con SGD            |
| `predict <file>`                       | Esegue inferenza su un file            |
| `save <file>`                          | Salva il modello                       |
| `load <file>`                          | Carica un modello salvato              |

---

## 💡 Caratteristiche principali

- Linguaggio DSL `.nectly` semplice e leggibile
- Modelli di rete neurale MLP (densamente connessi)
- Operazioni su GPU (CUDA) ottimizzate (matmul, batching)
- Addestramento su CPU con **SGD** e **MSE**
- Modello di esecuzione lineare e leggibile
- Nessuna dipendenza da Python o librerie esterne

---

## 📁 Struttura del progetto

```
nectly/
├── dataset.txt
├── example.nectly
├── nect.exe
├── makefile
├── kernels.cu
└── src/
    ├── main.c
    ├── parser.c/.h
    ├── tokenizer.c/.h
    ├── model.c/.h
    ├── optimizer.c/.h
    ├── loss.c/.h
    ├── util.c/.h
    ├── io_helpers.c/.h
    ├── gpu_helpers.c/.h
```

---

## 🔭 Roadmap

- ✅ Parsing e compilazione DSL `.nectly`
- ✅ Addestramento su CPU (SGD)
- ✅ Matmul CUDA con batching
- ⏳ Integrazione `predict_model` completo
- ⏳ Ottimizzatori avanzati: Adam, RMSprop
- ⏳ Supporto a RNN, LSTM
- ⏳ Plugin dinamici C per estensioni

---

## 📜 Licenza

Rilasciato sotto licenza **MIT**. Libero per uso personale, accademico e commerciale.

---

Sviluppato con ❤️ da [**jim871**](https://github.com/jim871) usando **C** e **CUDA**.

Per feedback, apri una issue o una pull request nel repository GitHub.




