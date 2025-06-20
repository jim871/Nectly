# 🚀 NECTLY – Neural C Training Language with CUDA Support

**NECTLY** è un linguaggio di programmazione standalone scritto in **C puro** con supporto a **CUDA**, progettato per **definire, addestrare e inferire modelli di deep learning** direttamente, senza dipendenze esterne.

> **Nota legale**: Il nome "NECT" usato in abbreviazione non è collegato in alcun modo al software di riconoscimento biometrico NECT GmbH, con sede in Germania.

## 📦 Requisiti minimi

- **Sistema operativo**: Windows 10/11 (x64)
- **Visual Studio 2022** (con C++ e toolchain MSVC)
- **CUDA Toolkit** 12.0 o superiore
- **GPU NVIDIA** compatibile (Compute Capability ≥ 5.0)
- (Opzionale) **MSYS2** o **WSL** per `make`

## ⚙️ Installazione (Windows)

### 1. Installa gli strumenti necessari

- [Visual Studio 2022](https://visualstudio.microsoft.com/) con componenti per C++
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- *(Facoltativo)* [MSYS2](https://www.msys2.org/) per `make`

### 2. Clona il repository

```bash
git clone https://github.com/TUO-USERNAME/nectly.git
cd nectly
3. Compilazione manuale (senza make)
Apri "Developer Command Prompt for VS 2022", poi:

bat
Copia
Modifica
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
set PATH=%CUDA_PATH%\bin;%PATH%
set LIB=%CUDA_PATH%\lib\x64;%LIB%
set INCLUDE=%CUDA_PATH%\include;%INCLUDE%

cl /c /MD /O2 /I src src\*.c main.c
nvcc -c -O2 -ccbin "cl" -Xcompiler /MD -I src src\kernels.cu -o kernels.obj
cl *.obj /link /OUT:nect.exe /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib
🧪 Esempio di utilizzo
File dataset.txt
yaml
Copia
Modifica
0.1 0.2 0.3 : 1.0
0.4 0.5 0.6 : 0.0
0.7 0.8 0.9 : 1.0
File example.nectly
python
Copia
Modifica
model Demo
input 3
layer 4
layer 1
train dataset.txt epochs 100 lr 0.01
predict dataset.txt
save model.bin
load model.bin
predict dataset.txt
Esegui:
bash
Copia
Modifica
nect.exe example.nectly
📖 Sintassi .nectly
Comando	Descrizione
model <nome>	Crea un modello
input <N>	Imposta dimensione input
layer <neuroni>	Aggiunge un layer denso
train <file> epochs <N> lr <f>	Addestra il modello con SGD
predict <file>	Esegue inferenza
save <file>	Salva il modello su disco
load <file>	Carica un modello da disco

💡 Caratteristiche principali
Linguaggio DSL semplificato (.nectly)

Definizione e training di reti neurali dense (MLP)

Supporto CUDA per operazioni su GPU (matmul, batching)

Addestramento su CPU (SGD, MSE)

Salvataggio/caricamento modelli

Nessuna dipendenza Python o framework esterni

📁 Struttura del progetto
bash
Copia
Modifica
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
🔭 Roadmap
 Implementazione predict_model completa

 Supporto per nuovi ottimizzatori (Adam, RMSprop)

 Salvataggio binario reale e formato compatibile

 Supporto a reti ricorrenti (RNN/LSTM)

 Estensioni modulari tramite plugin C dinamici

📜 Licenza
Rilasciato sotto licenza MIT. Libero per uso personale, accademico e commerciale.

Sviluppato con ❤️ da [TUO NOME] usando C + CUDA.

Per feedback, apri una issue o crea una pull request su GitHub.
















