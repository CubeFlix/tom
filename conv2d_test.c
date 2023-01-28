// conv2d_test.c

#include <stdio.h>
#include "headers.h"

void main() {
    random_init();

    int kernel_size = 2, channels = 3, filters = 2, samples = 5, height = 6, width = 6, out_height = 3, out_width = 3, stride = 2;
    struct matrix input, output, d_output, d_input;
    struct layer_conv2d c;

    matrix_init(&input, samples, channels * height * width);
    matrix_init(&output, samples, filters * out_height * out_width);
    matrix_init(&d_output, samples, filters * out_height * out_width);
    matrix_init(&d_input, samples, channels * height * width);
    layer_conv2d_init(&c, channels, height, width, filters, kernel_size, stride, &input, &output, &d_output, &d_input);
    layer_conv2d_init_values(&c, WI_ONES, BI_ONES);

    for (int i = 0; i < input.size; i++) {
        input.buffer[i] = 1.0;
    }
    for (int i = 0; i < d_output.size; i++) {
        d_output.buffer[i] = 1.0;
    }

    layer_conv2d_forward(&c);

    layer_conv2d_backward(&c);

    // for (int sample = 0; sample < samples; sample++) {
    //     for (int filter = 0; filter < filters; filter++) {
    //         for (int i = 0; i < out_height; i++) {
    //             for (int j = 0; j < out_width; j++) {
    //                 printf("%f ", output.buffer[sample * filters * out_height * out_width + filter * out_height * out_width + i * out_width + j]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //         }
    //     printf("\n");
    // }

    for (int filter = 0; filter < filters; filter++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int i = 0; i < kernel_size; i++) {
                for (int j = 0; j < kernel_size; j++) {
                    printf("%f ", c.d_weights.buffer[filter * channels * kernel_size * kernel_size + channel * kernel_size * kernel_size + i * kernel_size + j]);
                }
                printf("\n");
            }
            printf("\n");
            }
        printf("\n");
    }

    for (int filter = 0; filter < filters; filter++) {
        printf("%f ", c.d_biases.buffer[filter]);
    }

    layer_conv2d_free(&c);
    matrix_free(&input);
    matrix_free(&output);
    matrix_free(&d_output);
    matrix_free(&d_input);
    printf(LAST_ERROR);
    fflush(stdout);
}
