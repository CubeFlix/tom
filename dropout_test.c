// dropout_test.c

#include <stdio.h>
#include "headers.h"

void main() {
    int size = 5, samples = 10;
    struct matrix input, output, d_output, d_input;
    struct layer_dropout d;

    matrix_init(&input, samples, size);
    matrix_init(&output, samples, size);
    matrix_init(&d_output, samples, size);
    matrix_init(&d_input, samples, size);
    layer_dropout_init(&d, size, 0.1, &input, &output, &d_output, &d_input);

    for (int i = 0; i < input.size; i++) {
        input.buffer[i] = 1.0;
    }

    layer_dropout_forward(&d);

    layer_dropout_backward(&d);

    for (int i = 0; i < output.size; i++) {
        printf("%f ", output.buffer[i]);
    }

    layer_dropout_free(&d);
    matrix_free(&input);
    matrix_free(&output);
    matrix_free(&d_output);
    matrix_free(&d_input);
    printf("hello");
    fflush(stdout);
}
