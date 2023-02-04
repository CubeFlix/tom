// mnist_conv_test.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "headers.h"

// error handling macro
#define QUIT_ON_ERROR(x) { \
    int ret = (x); \
    if (!ret) { \
        printf("%s\n", LAST_ERROR); \
        exit(1); \
    } \
}

static void shuffle(void* X, void* Y, size_t n, size_t size_x, size_t size_y) {
    char tmp_x[size_x];
    char tmp_y[size_y];
    size_t stride_x = size_x * sizeof(char), stride_y = size_y * sizeof(char);

    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; ++i) {
            size_t rnd = (size_t)rand();
            size_t j = i + rnd / (RAND_MAX / (n - i) + 1);

            memcpy(tmp_x, X + j * stride_x, size_x);
            memcpy(X + j * stride_x, X + i * stride_x, size_x);
            memcpy(X + i * stride_x, tmp_x, size_x);

            memcpy(tmp_y, Y + j * stride_y, size_y);
            memcpy(Y + j * stride_y, Y + i * stride_y, size_y);
            memcpy(Y + i * stride_y, tmp_y, size_y);
        }
    }
}

void load_dataset(struct matrix* X, struct matrix* Y) {
    FILE* images = fopen("train-images-idx3-ubyte", "rb");
    FILE* labels = fopen("train-labels-idx1-ubyte", "rb");

    fseek(images, 16, 1);
    fseek(labels, 8, 1);

    unsigned char image[28 * 28];
    unsigned char label[1];

    for (int current = 0; current < 60000; current++) {
        fread((void*)image, 28 * 28, 1, images);
        fread((void*)label, 1, 1, labels);
        // Load the image into the matrix.
        for (int i = 0; i < 28 * 28; i++) {
            X->buffer[current * 28 * 28 + i] = (double)(image[i]) / 255.0;
        }

        // Load the label into the matrix.
        for (int i = 0; i < 10; i++) {
            if (i == label[0]) {
                Y->buffer[current * 10 + i] = 1.0;
            }
            else {
                Y->buffer[current * 10 + i] = 0.0;
            }
        }
    }
    fclose(images);
    fclose(labels);
}

void load_validation_dataset(struct matrix* X, struct matrix* Y) {
    FILE* images = fopen("t10k-images-idx3-ubyte", "rb");
    FILE* labels = fopen("t10k-labels-idx1-ubyte", "rb");

    fseek(images, 16, 1);
    fseek(labels, 8, 1);

    unsigned char image[28 * 28];
    unsigned char label[1];

    for (int current = 0; current < 10000; current++) {
        fread((void*)image, 28 * 28, 1, images);
        fread((void*)label, 1, 1, labels);
        // Load the image into the matrix.
        for (int i = 0; i < 28 * 28; i++) {
            X->buffer[current * 28 * 28 + i] = (double)(image[i]) / 255.0;
        }

        // Load the label into the matrix.
        for (int i = 0; i < 10; i++) {
            if (i == label[0]) {
                Y->buffer[current * 10 + i] = 1.0;
            }
            else {
                Y->buffer[current * 10 + i] = 0.0;
            }
        }
    }
    fclose(images);
    fclose(labels);
}

int main() {
    // Initialize RNG.
    random_init();

    // Initialize the network and its matrices.
    int data_size = 60000;
    int batch_size = 200;
    int input_size = 28 * 28;
    int c1_filters = 32;
    int c1_kern = 3;
    int c1_size = CALC_CONV2D_OUTPUT_DIM(28, 3, 1);
    int m1_size = CALC_MAXPOOL2D_OUTPUT_DIM(c1_size, 2, 2);

    // Prepare the training data.
    struct matrix X, Y;
    matrix_init(&X, data_size, input_size);
    matrix_init(&Y, data_size, 10);
    load_dataset(&X, &Y);
    shuffle(X.buffer, Y.buffer, data_size, sizeof(double) * input_size, sizeof(double) * 10);

    // Create the model.
    struct model* m = calloc(1, sizeof(struct model));
    model_init(m, batch_size);
    struct layer* c1 = model_add_conv2d_layer(m, 1, 28, 28, c1_filters, c1_kern, 1); // 28 * 28 -> 32 * 
    struct layer* a1 = model_add_layer(m, LAYER_RELU, c1_size * c1_size * c1_filters, c1_size * c1_size * c1_filters);
    struct layer* m1 = model_add_maxpool2d_layer(m, c1_filters, c1_size, c1_size, 2, 2);
    struct layer* l1 = model_add_layer(m, LAYER_DENSE, c1_filters * m1_size * m1_size, 100);
    struct layer* a2 = model_add_layer(m, LAYER_RELU, 100, 100);
    struct layer* l2 = model_add_layer(m, LAYER_DENSE, 100, 10);
    struct layer* a3 = model_add_layer(m, LAYER_SOFTMAX, 10, 10);
    model_set_loss(m, LOSS_CROSSENTROPY);
    QUIT_ON_ERROR(model_finalize(m));
    layer_conv2d_init_values(c1->obj, WI_HE_NORMAL, BI_ZEROS);
    layer_dense_init_values(l1->obj, WI_HE_NORMAL, BI_ZEROS);
    layer_dense_init_values(l2->obj, WI_HE_NORMAL, BI_ZEROS);
    QUIT_ON_ERROR(model_init_optimizers(m, OPTIMIZER_ADAM, 0.001, 0.9, 0.999, 0.0, 1.0e-7));

    // Train the network.
    printf("training...\n");
    QUIT_ON_ERROR(model_train(m, &X, &Y, 2, true));

    // Print the final output.
    data_size = 10000;
    matrix_free(&X);
    matrix_free(&Y);
    matrix_init(&X, data_size, input_size);
    matrix_init(&Y, data_size, 10);
    load_validation_dataset(&X, &Y);
    shuffle(X.buffer, Y.buffer, data_size, sizeof(double) * input_size, sizeof(double) * 10);
    double val_loss = model_calc_loss(m, &X, &Y);
    printf("validation loss: %f\n", val_loss);

    matrix_free(&X);
    matrix_free(&Y);
    model_free(m);
    free(m);
}
