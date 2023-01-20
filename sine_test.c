// sine_test.c

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "dense.h"
#include "matrix.h"
#include "errors.h"
#include "random.h"
#include "relu.h"
#include "mse.h"
#include "adam.h"

static void shuffle(void *X, void *Y, size_t n, size_t size) {
    char tmp[size];
    size_t stride = size * sizeof(char);

    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; ++i) {
            size_t rnd = (size_t) rand();
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

void main() {
    // Initialize RNG.
    random_init();

    // Initialize the network and its matrices.
    int data_size = 200;
    int batch_size = 25;
    int input_size = 1;
    int h1_size = 16;
    int h2_size = 16;
    int h3_size = 1;

    struct layer_dense h1, h2, h3;
    struct activation_relu a1, a2;
    struct loss_mse l;

    // Prepare the training data.
    struct matrix X, Y;
    matrix_init(&X, data_size, input_size);
    matrix_init(&Y, data_size, h3_size);
    for (int i = 0; i < X.size; i++) {
        X.buffer[i] = (double)i / (double)X.size;
    }
    for (int i = 0; i < Y.size; i++) {
        Y.buffer[i] = (sin(X.buffer[i] * 6.0) + 1.0) / 2.0;
    }
    shuffle(X.buffer, Y.buffer, data_size, sizeof(double));

    struct matrix input, h1_output, a1_output, h2_output, a2_output, h3_output, l_output, y;
    matrix_init(&input, batch_size, input_size);
    matrix_init(&h1_output, batch_size, h1_size);
    matrix_init(&a1_output, batch_size, h1_size);
    matrix_init(&h2_output, batch_size, h2_size);
    matrix_init(&a2_output, batch_size, h2_size);
    matrix_init(&h3_output, batch_size, h3_size);
    matrix_init(&l_output, batch_size, 1);
    matrix_init(&y, batch_size, h3_size);

    struct matrix l_d_inputs, h3_d_inputs, a2_d_inputs, h2_d_inputs, a1_d_inputs, h1_d_inputs;
    matrix_init(&l_d_inputs, batch_size, h3_size);
    matrix_init(&h3_d_inputs, batch_size, h2_size);
    matrix_init(&a2_d_inputs, batch_size, h2_size);
    matrix_init(&h2_d_inputs, batch_size, h1_size);
    matrix_init(&a1_d_inputs, batch_size, h1_size);
    matrix_init(&h1_d_inputs, batch_size, input_size);

    layer_dense_init(&h1, input_size, h1_size, &input, &h1_output, &a1_d_inputs, &h1_d_inputs);
    layer_dense_init_values(&h1, WI_GLOROT_NORMAL, BI_ZEROS);
    activation_relu_init(&a1, h1_size, &h1_output, &a1_output, &h2_d_inputs, &a1_d_inputs);
    layer_dense_init(&h2, h1_size, h2_size, &a1_output, &h2_output, &a2_d_inputs, &h2_d_inputs);
    layer_dense_init_values(&h2, WI_GLOROT_NORMAL, BI_ZEROS);
    activation_relu_init(&a2, h2_size, &h2_output, &a2_output, &h3_d_inputs, &a2_d_inputs);
    layer_dense_init(&h3, h2_size, h3_size, &a2_output, &h3_output, &l_d_inputs, &h3_d_inputs);
    layer_dense_init_values(&h3, WI_GLOROT_NORMAL, BI_ZEROS);
    loss_mse_init(&l, h3_size, &h3_output, &y, &l_output, &l_d_inputs);

    struct optimizer_adam adam_h1, adam_h2, adam_h3;
    double learning_rate = 0.001;
    optimizer_adam_init(&adam_h1, &h1, learning_rate, 0.9, 0.999, 0, 1.0e-7);
    optimizer_adam_init(&adam_h2, &h2, learning_rate, 0.9, 0.999, 0, 1.0e-7);
    optimizer_adam_init(&adam_h3, &h3, learning_rate, 0.9, 0.999, 0, 1.0e-7);

    // Train the network.
    double loss;
    for (int epoch = 0; epoch < 1000; epoch++) {
        for (int batch = 0; batch < data_size; batch += batch_size) {
            // Load in the batch data.
            memcpy((void*)input.buffer, (void*)&X.buffer[batch], batch_size * sizeof(double));
            memcpy((void*)y.buffer, (void*)&Y.buffer[batch], batch_size * sizeof(double));

            // Perform a forward pass.
            layer_dense_forward(&h1);
            activation_relu_forward(&a1);
            layer_dense_forward(&h2);
            activation_relu_forward(&a2);
            layer_dense_forward(&h3);
            loss = loss_mse_forward(&l);

            // Perform a backward pass.
            loss_mse_backward(&l);
            layer_dense_backward(&h3);
            activation_relu_backward(&a2);
            layer_dense_backward(&h2);
            activation_relu_backward(&a1);
            layer_dense_backward(&h1);

            // Train.
            optimizer_adam_update(&adam_h1, epoch);
            optimizer_adam_update(&adam_h2, epoch);
            optimizer_adam_update(&adam_h3, epoch);
        }
        // printf("epoch: %d, loss %f\n", epoch, loss);
    }

    // Print the final output.
    for (int batch = 0; batch < data_size; batch += batch_size) {
        // Load in the batch data.
        memcpy((void*)input.buffer, (void*)&X.buffer[batch], batch_size * sizeof(double));
        memcpy((void*)y.buffer, (void*)&Y.buffer[batch], batch_size * sizeof(double));
        
        // Perform a forward pass.
        layer_dense_forward(&h1);
        activation_relu_forward(&a1);
        layer_dense_forward(&h2);
        activation_relu_forward(&a2);
        layer_dense_forward(&h3);
        
        for (int i = 0; i < y.size; i++) {
            printf("%f, %f, %f\n", input.buffer[i], y.buffer[i], h3_output.buffer[i]);
        }
    }

    // Save the network.
    FILE *save_file = fopen("sine_test.dat", "w");
    if (save_file != NULL) {
        fwrite(h1.weights.buffer, h1.weights.size * sizeof(double), 1, save_file);
        fwrite(h1.biases.buffer, h1.biases.size * sizeof(double), 1, save_file);
        fwrite(h2.weights.buffer, h2.weights.size * sizeof(double), 1, save_file);
        fwrite(h2.biases.buffer, h2.biases.size * sizeof(double), 1, save_file);
        fwrite(h3.weights.buffer, h3.weights.size * sizeof(double), 1, save_file);
        fwrite(h3.biases.buffer, h3.biases.size * sizeof(double), 1, save_file);
    } else {
        printf("Failed to save sine_test network.\n");
    }
    fclose(save_file);

    // Free the network and its matrices.
    layer_dense_free(&h1);
    layer_dense_free(&h2);
    layer_dense_free(&h3);
    optimizer_adam_free(&adam_h1);
    optimizer_adam_free(&adam_h2);
    optimizer_adam_free(&adam_h3);
    matrix_free(&X);
    matrix_free(&Y);
    matrix_free(&input);
    matrix_free(&h1_output);
    matrix_free(&a1_output);
    matrix_free(&h2_output);
    matrix_free(&a2_output);
    matrix_free(&h3_output);
    matrix_free(&l_output);
    matrix_free(&y);
    matrix_free(&l_d_inputs);
    matrix_free(&h3_d_inputs);
    matrix_free(&a2_d_inputs);
    matrix_free(&h2_d_inputs);
    matrix_free(&a1_d_inputs);
    matrix_free(&h1_d_inputs);
}