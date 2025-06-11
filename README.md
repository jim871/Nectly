# NECT GPU Language

NECT è un **linguaggio di programmazione standalone** sviluppato in **C puro** con supporto **CUDA** per definire, addestrare e inferire modelli di deep learning di qualsiasi dimensione.

---

## 📖 Panoramica

NECT offre una **DSL semplice e leggibile** (`.nect`) per:

* Definire la **struttura** di reti neurali feedforward dinamiche
* Eseguire il **forward pass** su GPU tramite kernel CUDA
* Eseguire il **backward pass** (SGD, MSE loss) su CPU
* Salvare e caricare modelli in formato binario
* Provare rapidamente esempi con **zero dipendenze Python**

L’intero runtime è scritto in C, senza librerie esterne se non la **CUDA Runtime**.

---

## ⚙️ Caratteristiche principali

* **Sintassi DSL** `.nect` intuitiva e priva di boilerplate.
* **Modelli dinamici**: nessun limite fisso a numero di layer o neuroni.
* **GPU-accelerazione**: forward pass svolto con matrici 16×16 ottimizzate.
* **Training reale**: backward pass con **SGD** e **MSE loss**.
* **Salvataggio/Caricamento** di modelli di qualsiasi complessità.
* **Makefile unificato**: compila con `gcc` e `nvcc`.
* **Esempi predefiniti** per partire subito.

---

## 📋 Requisiti di sistema

* **Sistema Linux** (o macchine con toolchain GCC e NVCC)
* **GCC** (>= 9.0) con supporto C11
* **NVCC** e **CUDA Toolkit** (>= 11.0)
* Almeno **4 GB di RAM** per modelli di piccola scala
* GPU NVIDIA con compute capability >= 5.0

---

## 🚀 Installazione

1. **Clona il repository**

   ```bash
   git clone https://github.com/tuoutente/nect-gpu.git
   cd nect-gpu
   ```

2. **Compila tutto**

   ```bash
   make
   ```

   * Genererà l’eseguibile `nect` e l’oggetto CUDA `kernels.o`.

3. **Verifica**

   ```bash
   ./nect example.nect
   ```

   Dovresti vedere log di inizializzazione, training e predizioni.

---

## 📝 Sintassi DSL (.nect)

Tutti i comandi sono su righe separate:

| Comando                          | Descrizione                                                         |
| -------------------------------- | ------------------------------------------------------------------- |
| `model <nome>`                   | Inizializza un nuovo modello                                        |
| `input <dimensione>`             | Imposta la dimensione del vettore di input                          |
| `layer <unità>`                  | Aggiunge un layer denso con \<unità> neuroni                        |
| `train <file> epochs <n> lr <f>` | Avvia training su `<file>` per `<n>` epoche con learning rate `<f>` |
| `predict <file>`                 | Esegue inferenza su `<file>`                                        |
| `save <file>`                    | Salva il modello corrente in `<file>`                               |
| `load <file>`                    | Carica da `<file>` un modello precedentemente salvato               |

---

### Esempio di script `example.nect`

```nect
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

---

## 🏗️ Architettura interna

1. **Parser** (`src/parser.c`) legge e smista comandi.
2. **Modello dinamico** (`src/model.c`): struttura con `malloc/realloc` per layer, pesi e bias.
3. **Forward pass**:

   * Input e pesi trasferiti su GPU
   * Kernel CUDA (`kernels.cu`) esegue la moltiplicazione matriciale
   * Risultati copiati indietro in memoria host
4. **Backward pass** (CPU): calcolo del gradiente MSE e aggiornamento SGD.
5. **IO**: salvataggio e caricamento binario di pesi e topologia.

---

## 📂 Struttura dei file

```
./
├── LICENSE
├── README.md
├── Makefile
├── kernels.cu
├── example.nect
├── dataset.txt
└── src/
    ├── main.c
    ├── parser.h
    ├── parser.c
    ├── model.h
    ├── model.c
    ├── util.h
    ├── util.c
    ├── gpu_helpers.h
    └── gpu_helpers.c
```

---

## 🔧 Estensioni e personalizzazioni

* **Attivazioni**: integra ReLU/GELU nei layer.
* **Ottimizzatori**: aggiungi Adam, RMSProp, ecc.
* **Loss**: supporta cross-entropy o altri criteri.
* **Multi-GPU**: espandi il runtime per bilanciare carico.
* **Tokenizer**: costruisci modelli di NLP integrando BPE.

---

## 👥 Contribuire

1. Fork del progetto
2. Crea un branch di funzionalità
3. Effettua commit e push
4. Apri una pull request descrivendo le modifiche

---

## 📄 Licenza

Distribuito sotto **MIT License**. Vedi il file `LICENSE` per i dettagli.

---

*Buon divertimento con NECT GPU!*
