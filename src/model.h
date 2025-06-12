#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Initialize a new model with given name. */
int init_model(const char *name);

/** Set the input dimension for the model. */
int set_input_dim(int dim);

/** Add a dense layer with given number of units. */
int add_layer(int units);

/**
 * Train the model on dataset file using multiple GPUs.
 * @param path Path to dataset
 * @param epochs Number of epochs
 * @param lr Learning rate
 * @param nGPUs Number of CUDA devices available
 * @param streams Array of CUDA streams, one per device
 */
int train_model(const char *path, int epochs, float lr, int nGPUs, cudaStream_t *streams);

/**
 * Run inference on dataset file using multiple GPUs.
 * @param path Path to dataset
 * @param nGPUs Number of CUDA devices available
 * @param streams Array of CUDA streams, one per device
 */
int predict_model(const char *path, int nGPUs, cudaStream_t *streams);

/** Save the current model to file. */
int save_model(const char *path);

/** Load a model from file. */
int load_model(const char *path);

#ifdef __cplusplus
}
#endif
