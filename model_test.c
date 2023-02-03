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
#include <string.h>

#include "headers.h"

void segvHandler(int s) {
    printf("Segmentation Fault %d\n", s);
    exit(1);
}

void abrtHandler(int s) {
    printf("Aborted %d\n", s);
    exit(1);
}

static void shuffle(void* X, void* Y, size_t n, size_t size) {
    char tmp[size];
    size_t stride = size * sizeof(char);

    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; ++i) {
            size_t rnd = (size_t)rand();
            size_t j = i + rnd / (RAND_MAX / (n - i) + 1);

            memcpy(tmp, X + j * stride, size);
            memcpy(X + j * stride, X + i * stride, size);
            memcpy(X + i * stride, tmp, size);

            memcpy(tmp, Y + j * stride, size);
            memcpy(Y + j * stride, Y + i * stride, size);
            memcpy(Y + i * stride, tmp, size);
        }
    }
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
    shuffle(X.buffer, Y.buffer, n_samples, sizeof(double));

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
    QUIT_ON_ERROR(model_init_optimizers(m, OPTIMIZER_ADAM, 0.001, 0.9, 0.999, 0.0, 1e-7));
    
    // Train the model.
    QUIT_ON_ERROR(model_train(m, &X, &Y, 1000, false));
    double loss = model_calc_loss(m, &X, &Y);
    printf("loss: %f\n", loss);

    // Model output values.
    struct matrix yhat;
    QUIT_ON_ERROR(matrix_init(&yhat, n_samples, 1));
    QUIT_ON_ERROR(model_predict(m, &X, &yhat));
    for (int i = 0; i < n_samples; i++) {
        printf("%f, %f, %f\n", X.buffer[i], Y.buffer[i], yhat.buffer[i]);
    }

    model_free(m);
    free(m);
    matrix_free(&X);
    matrix_free(&Y);
    matrix_free(&yhat);
    printf("done\n");
    return 0;
}