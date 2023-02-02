// model_test.c

// error handling macro
#define QUIT_ON_ERROR(x) { \
    int ret = (x); \
    if (!ret) { \
        printf("%s\n", LAST_ERROR); \
        exit(1); \
    } \
}

// segfault handling
#include <signal.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "headers.h"

void segvHandler(int s) {
    printf("Segmentation Fault %d\n", s);
    exit(1);
}

void abrtHandler(int s) {
    printf("Aborted %d\n", s);
    exit(1);
}

int main(void) {
    signal(SIGSEGV, segvHandler);
    signal(SIGABRT, abrtHandler);
    random_init();

    // Prepare the data.
    struct matrix X, Y;
    int n_samples = 500;
    QUIT_ON_ERROR(matrix_init(&X, n_samples, 1));
    QUIT_ON_ERROR(matrix_init(&Y, n_samples, 1));
    for (int i = 0; i < n_samples; i++) {
        X.buffer[i] = (double)i / (double)n_samples;
        Y.buffer[i] = sin(X.buffer[i] * 5.0);
    }

    // Create the model.
    struct model *m = calloc(1, sizeof(struct model));
    model_init(m, 100);
    struct layer* l1 = model_add_layer(m, LAYER_DENSE, 1, 64);
    struct layer* a1 = model_add_layer(m, LAYER_RELU, 64, 64);
    struct layer* l2 = model_add_layer(m, LAYER_DENSE, 64, 1);
    model_set_loss(m, LOSS_MSE);
    QUIT_ON_ERROR(model_finalize(m));
    layer_dense_init_values(l1->obj, WI_GLOROT_NORMAL, BI_ZEROS);
    layer_dense_init_values(l2->obj, WI_GLOROT_NORMAL, BI_ZEROS);
    QUIT_ON_ERROR(model_init_optimizers(m, OPTIMIZER_SGD, 0.001, 0.0, 0.0));
    
    // Train the model.
    QUIT_ON_ERROR(model_train(m, &X, &Y, 1000, false));
    double loss = model_calc_loss(m, &X, &Y);
    printf("loss: %f\n", loss);

    model_free(m);
    free(m);
    matrix_free(&X);
    matrix_free(&Y);
    printf("done\n");
    return 0;
}