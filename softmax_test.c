// softmax_test.c

#include <stdio.h>
#include "matrix.h"
#include "errors.h"
#include "softmax.h"
#include "crossentropy.h"

void main() {
    int size = 5, samples = 10;
    struct matrix input, y, a1_output, l_output, l_d_input, a1_d_input;
    struct activation_softmax a1;
    struct loss_crossentropy l;

    matrix_init(&input, samples, size);
    matrix_init(&y, samples, size);
    matrix_init(&a1_output, samples, size);
    matrix_init(&l_output, samples, 1);
    matrix_init(&l_d_input, samples, size);
    matrix_init(&a1_d_input, samples, size);
    activation_softmax_init(&a1, size, &input, &a1_output, &l_d_input, &a1_d_input);
    loss_crossentropy_init(&l, size, &a1_output, &y, &l_output, &l_d_input);

    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < size; j++) {
            input.buffer[i * size + j] = (double)j / (double)size;
            if (j == 2) {
                y.buffer[i * size + j] = 1.0;
            } else {
                y.buffer[i * size + j] = 0.0;
            }
        }
    }

    activation_softmax_forward_stable(&a1);
    loss_crossentropy_forward(&l);

    loss_crossentropy_backward_softmax(&l);
    // activation_softmax_backward(&a1);

    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < size; j++) {
            printf("%f ", l_d_input.buffer[i * size + j]);
        }
        printf("\n");
    }

    activation_softmax_free(&a1);
    matrix_free(&input);
    matrix_free(&y);
    matrix_free(&a1_output);
    matrix_free(&l_output);
    matrix_free(&l_d_input);
    matrix_free(&a1_d_input);
    printf("hello");
    fflush(stdout);
}
