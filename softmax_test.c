// softmax_test.c

#include <stdio.h>
#include "matrix.h"
#include "errors.h"
#include "softmax.h"

void main() {
    int size = 5, samples = 10;
    struct matrix input, output, d_output, d_input;
    struct activation_softmax a1;

    matrix_init(&input, samples, size);
    matrix_init(&output, samples, size);
    matrix_init(&d_output, samples, size);
    matrix_init(&d_input, samples, size);
    activation_softmax_init(&a1, size, &input, &output, &d_output, &d_input);

    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < size; j++) {
            input.buffer[i * size + j] = (double)j / (double)size;
        }
    }

    activation_softmax_forward_stable(&a1);

    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < size; j++) {
            d_output.buffer[i * size + j] = output.buffer[i * size + j];
        }
    }

    activation_softmax_backward(&a1);

    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < size; j++) {
            printf("%f ", d_input.buffer[i * size + j]);
        }
        printf("\n");
    }

    activation_softmax_free(&a1);
    matrix_free(&input);
    matrix_free(&output);
    matrix_free(&d_output);
    matrix_free(&d_input);
    printf("hello");
    fflush(stdout);
}
