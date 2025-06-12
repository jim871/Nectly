#pragma once
int init_model(const char *name);
int set_input_dim(int dim);
int add_layer(int units);
int train_model(const char *path, int epochs, float lr);
int predict_model(const char *path);
int save_model(const char *path);
int load_model(const char *path);
