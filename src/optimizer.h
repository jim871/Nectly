void optimizer_step(
    float **weights,
    float **grads,
    float **moments,
    int     *layer_sizes,
    size_t   n_layers,
    const float *ce_grads,
    int      seq_len,
    int      vocab_size,
    float    lr,
    float    beta1,
    float    beta2,
    float    eps,
    float    wd,
    int     *t_step
);
