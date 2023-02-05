// mnist_test.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "matrix.h"
#include "dense.h"
#include "relu.h"
#include "softmax.h"
#include "crossentropy.h"
#include "random.h"
#include "adam.h"
#include "errors.h"

static void shuffle(void *X, void *Y, size_t n, size_t size_x, size_t size_y) {
    char tmp_x[size_x];
    char tmp_y[size_y];
    size_t stride_x = size_x * sizeof(char), stride_y = size_y * sizeof(char);

    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; ++i) {
            size_t rnd = (size_t) rand();
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

void load_dataset(struct matrix *X, struct matrix *Y) {
    FILE* images = fopen("train-images-idx3-ubyte", "rb");
    FILE* labels = fopen("train-labels-idx1-ubyte", "rb");

    fseek(images, 16, 1);
    fseek(labels, 8, 1);
    
    unsigned char image[28 * 28];
    unsigned char label[1];

    for (int current = 0; current < 60000; current++) {
        fread((void*)image, 28 * 28, 1, images);
        fread((void*)label,  1, 1, labels);
        // Load the image into the matrix.
        for (int i = 0; i < 28*28; i++) {
            X->buffer[current * 28*28 + i] = (double)(image[i]) / 255.0;
        }

        // Load the label into the matrix.
        for (int i = 0; i < 10; i++) {
            if (i == label[0]) {
                Y->buffer[current * 10 + i] = 1.0;
            } else {
                Y->buffer[current * 10 + i] = 0.0;
            }
        }
    }
    fclose(images);
    fclose(labels);
}

void load_validation_dataset(struct matrix *X, struct matrix *Y) {
    FILE* images = fopen("t10k-images-idx3-ubyte", "rb");
    FILE* labels = fopen("t10k-labels-idx1-ubyte", "rb");

    fseek(images, 16, 1);
    fseek(labels, 8, 1);
    
    unsigned char image[28 * 28];
    unsigned char label[1];

    for (int current = 0; current < 10000; current++) {
        fread((void *)image, 28 * 28, 1, images);
        fread((void *)label, 1, 1, labels);
        // Load the image into the matrix.
        for (int i = 0; i < 28*28; i++) {
            X->buffer[current * 28*28 + i] = (double)(image[i]) / 255.0;
        }

        // Load the label into the matrix.
        for (int i = 0; i < 10; i++) {
            if (i == label[0]) {
                Y->buffer[current * 10 + i] = 1.0;
            } else {
                Y->buffer[current * 10 + i] = 0.0;
            }
        }
    }
    fclose(images);
    fclose(labels);
}

void load_network(char *path, void *h1w, void *h1b, void *h2w, void *h2b) {
	FILE* params = fopen(path, "rb");
	fread(h1w, 28 * 28 * 400 * sizeof(double), 1, params);
	fread(h1b, 400 * sizeof(double), 1, params);
	fread(h2w, 400 * 10 * sizeof(double),1,  params);
	fread(h2b, 10 * sizeof(double), 1, params);
	fclose(params);
}

int main() {
    // Initialize RNG.
    random_init();

    // Initialize the network and its matrices.
    int data_size = 60000;
    int batch_size = 200;
    int input_size = 28 * 28;
    int h1_size = 400;
    int h2_size = 10;

    struct layer_dense h1, h2;
    struct activation_relu a1;
    struct activation_softmax a2;
    struct loss_crossentropy l;
    
    // Prepare the training data.
    struct matrix X, Y;
    matrix_init(&X, data_size, input_size);
    matrix_init(&Y, data_size, h2_size);
    load_dataset(&X, &Y);

    struct matrix input, h1_output, a1_output, h2_output, a2_output, l_output, y;
    matrix_init(&input, batch_size, input_size);
    matrix_init(&h1_output, batch_size, h1_size);
    matrix_init(&a1_output, batch_size, h1_size);
    matrix_init(&h2_output, batch_size, h2_size);
    matrix_init(&a2_output, batch_size, h2_size);
    matrix_init(&l_output, batch_size, 1);
    matrix_init(&y, batch_size, h2_size);

    struct matrix l_d_inputs, h2_d_inputs, a1_d_inputs, h1_d_inputs;
    matrix_init(&l_d_inputs, batch_size, h2_size);
    matrix_init(&h2_d_inputs, batch_size, h1_size);
    matrix_init(&a1_d_inputs, batch_size, h1_size);
    matrix_init(&h1_d_inputs, batch_size, input_size);

    layer_dense_init(&h1, input_size, h1_size, &input, &h1_output, &a1_d_inputs, &h1_d_inputs);
    layer_dense_init_values(&h1, WI_HE_NORMAL, BI_ZEROS);
    // layer_dense_init_regularization(&h1, 0.000, 0.000, 0.00, 0.000);
    activation_relu_init(&a1, h1_size, &h1_output, &a1_output, &h2_d_inputs, &a1_d_inputs);
    layer_dense_init(&h2, h1_size, h2_size, &a1_output, &h2_output, &l_d_inputs, &h2_d_inputs);
    layer_dense_init_values(&h2, WI_HE_NORMAL, BI_ZEROS);
    // layer_dense_init_regularization(&h2, 0.000, 0.000, 0.0, 0.000);
    // Note that the softmax's gradients will not be used in training.
    activation_softmax_init(&a2, h2_size, &h2_output, &a2_output, &l_d_inputs, &l_d_inputs);
    loss_crossentropy_init(&l, h2_size, &a2_output, &y, &l_output, &l_d_inputs);

    struct optimizer_adam adam_h1, adam_h2;
    double learning_rate = 0.001;
    optimizer_adam_init(&adam_h1, &h1, learning_rate, 0.9, 0.999, 0, 1.0e-7);
    optimizer_adam_init(&adam_h2, &h2, learning_rate, 0.9, 0.999, 0, 1.0e-7);

    // Train the network.
    double loss;
    int debug_out_size;
    char debug_out[64];
    for (int epoch = 0; epoch < 2; epoch++) {
        shuffle(X.buffer, Y.buffer, data_size, sizeof(double) * input_size, sizeof(double) * h2_size);
        loss = 0.0;
        for (int batch = 0; batch < data_size; batch += batch_size) {
            // Load in the batch data.
            memcpy((void*)input.buffer, (void*)&X.buffer[batch * input_size], batch_size * input_size * sizeof(double));
            memcpy((void*)y.buffer, (void*)&Y.buffer[batch * h2_size], batch_size * h2_size * sizeof(double));

            // Perform a forward pass.
            layer_dense_forward(&h1);
            activation_relu_forward(&a1);
            layer_dense_forward(&h2);
            activation_softmax_forward_stable(&a2);
            loss += loss_crossentropy_forward(&l);

            // Perform a backward pass.
            loss_crossentropy_backward_softmax(&l);
            layer_dense_backward(&h2);
            activation_relu_backward(&a1);
            layer_dense_backward(&h1);

            // Train.
            optimizer_adam_update(&adam_h1, epoch * (data_size / batch_size) + (batch / batch_size));
            optimizer_adam_update(&adam_h2, epoch * (data_size / batch_size) + (batch / batch_size));

            debug_out_size = sprintf(debug_out, "\rbatch: %d/%d avg loss: %f", batch/batch_size, data_size/batch_size, loss / (double)(batch / batch_size));
            printf(debug_out);
            fflush(stdout);
        }
        printf("\n");
        printf("epoch: %d, loss %f\n", epoch, loss / (double)(data_size / batch_size));
    }

    // load_network("mnist_test.dat", (void*)h1.weights.buffer, (void*)h1.biases.buffer, (void*)h2.weights.buffer, (void*)h2.biases.buffer);

    // Print the final output.
    data_size = 10000;
    matrix_free(&X);
    matrix_free(&Y);
    matrix_init(&X, data_size, input_size);
    matrix_init(&Y, data_size, h2_size);
    load_validation_dataset(&X, &Y);
    shuffle(X.buffer, Y.buffer, data_size, sizeof(double) * input_size, sizeof(double) * h2_size);
    double max;
    int max_index = 0, max_y_index = 0, num_correct = 0;
    for (int batch = 0; batch < data_size; batch += batch_size) {
        // Load in the batch data.
        memcpy((void*)input.buffer, (void*)&X.buffer[batch * input_size], batch_size * input_size * sizeof(double));
        memcpy((void*)y.buffer, (void*)&Y.buffer[batch * h2_size], batch_size * h2_size * sizeof(double));
        
        // Perform a forward pass.
        layer_dense_forward(&h1);
        activation_relu_forward(&a1);
        layer_dense_forward(&h2);
        activation_softmax_forward_stable(&a2);
        
        for (int i = 0; i < y.n_rows; i++) {
            max = -1.0;
            for (int j = 0; j < y.n_cols; j++) {
                if (y.buffer[i * h2_size + j] > max) {
                    max = y.buffer[i * h2_size + j];
                    max_y_index = j;
                }
            }

            max = -DBL_MAX;
            for (int j = 0; j < y.n_cols; j++) {
                if (a2_output.buffer[i * h2_size + j] > max) {
                    max = a2_output.buffer[i * h2_size + j];
                    max_index = j;
                }
            }

            if (max_index == max_y_index) {
                num_correct += 1;
            }
        }
    }
    printf("validation accuracy: %d %d\n", num_correct, 10000);
    printf("validation accuracy: %f\n", (double)num_correct/10000.0);

    // Save the network.
    FILE *save_file = fopen("mnist_test.dat", "wb");
    if (save_file != NULL) {
        fwrite(h1.weights.buffer, h1.weights.size * sizeof(double), 1, save_file);
        fwrite(h1.biases.buffer, h1.biases.size * sizeof(double), 1, save_file);
        fwrite(h2.weights.buffer, h2.weights.size * sizeof(double), 1, save_file);
        fwrite(h2.biases.buffer, h2.biases.size * sizeof(double), 1, save_file);
    } else {
        printf("Failed to save mnist_test network.\n");
    }
    fclose(save_file);

    // Free the network and its matrices.
    layer_dense_free(&h1);
    layer_dense_free(&h2);
    optimizer_adam_free(&adam_h1);
    optimizer_adam_free(&adam_h2);
    matrix_free(&X);
    matrix_free(&Y);
    matrix_free(&input);
    matrix_free(&h1_output);
    matrix_free(&a1_output);
    matrix_free(&h2_output);
    matrix_free(&a2_output);
    matrix_free(&l_output);
    matrix_free(&y);
    matrix_free(&l_d_inputs);
    matrix_free(&h2_d_inputs);
    matrix_free(&a1_d_inputs);
    matrix_free(&h1_d_inputs);
}
