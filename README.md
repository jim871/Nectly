# NECT: Neural C Training Language with CUDA Support

NECT è un linguaggio di programmazione standalone sviluppato in **C puro** con supporto **CUDA**, progettato per definire, addestrare e inferire modelli di deep learning senza dipendenze esterne.
Nota legale: il comando Nect é usato solo per abbreviazione e non ha nulla a che vedere con il software Nect di riconoscimento di impronta digitale della azienda con sede in Germania
## 📅 Requisiti minimi

* **Sistema operativo**: Windows 10/11 (x64)
* **Visual Studio 2022** (con supporto C++)
* **CUDA Toolkit** 12.9 o superiore
* **GPU NVIDIA compatibile CUDA** (Compute Capability >= 5.0)
* **Make per Windows** (incluso in MSYS2 o WSL consigliato)

## ⚙️ Installazione passo-passo (Windows)

### 1. Installa gli strumenti necessari

* **Visual Studio 2022** con componenti C++
* **CUDA Toolkit** da [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
* **MSYS2** per usare `make` e shell POSIX-like ([https://www.msys2.org/](https://www.msys2.org/))

Dopo l'installazione, apri la shell **MSYS2 UCRT64**.

### 2. Clona il repository

```bash
cd ~/Desktop
git clone https://github.com/TUO-USERNAME/nect.git
cd nect
```

### 3. Compila il progetto

```bash
make
```

Se tutto va a buon fine, verrà generato un eseguibile chiamato `main.exe`.

## 🔧 Esecuzione di esempio

Assicurati di avere un file `dataset.txt` come questo:

```
0.1 0.2 0.3 : 1.0
0.4 0.5 0.6 : 0.0
0.7 0.8 0.9 : 1.0
```

E uno script `example.nect`:

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

Esegui:

```bash
./main.exe example.nect
```

## 📖 Sintassi DSL `.nect`

| Comando                          | Descrizione                        |
| -------------------------------- | ---------------------------------- |
| `model <nome>`                   | Crea un nuovo modello              |
| `input <N>`                      | Specifica la dimensione dell'input |
| `layer <neuroni>`                | Aggiunge un layer denso            |
| `train <file> epochs <N> lr <f>` | Addestra il modello                |
| `predict <file>`                 | Esegue inferenza                   |
| `save <file>`                    | Salva il modello                   |
| `load <file>`                    | Carica un modello salvato          |

## 🌌 Caratteristiche principali

* Reti neurali dinamiche (nessun limite a layer/neuroni)
* Forward pass su GPU (CUDA)
* Backpropagation su CPU (SGD + MSE loss)
* Salvataggio/caricamento modelli
* Nessuna dipendenza Python

## 🏠 Struttura del progetto

```
nect/
├── dataset.txt
├── example.nect
├── main.exe
├── makefile
├── kernels.cu
└── src/
    ├── main.c
    ├── parser.c/.h
    ├── model.c/.h
    ├── util.c/.h
    ├── gpu_helpers.c/.h
```

## 💪 Roadmap

*

## 🙏 Licenza

MIT License. Libero per uso personale e commerciale.

---

Sviluppato con ❤️ in C e CUDA.

> Per problemi o suggerimenti, apri una issue o una pull request su GitHub.
