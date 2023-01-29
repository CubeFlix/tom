// conv2d_test.c

#include <stdio.h>
#include "headers.h"

void main() {
    random_init();

    int kernel_size = 2, channels = 3, filters = 2, samples = 5, height = 6, width = 6, out_height = 3, out_width = 3, stride = 2;
    struct matrix input, output, pool_output, d_output, d_input, d_output_pool;
    struct layer_conv2d c;
    struct layer_maxpool2d m;

    matrix_init(&input, samples, channels * height * width);
    matrix_init(&output, samples, filters * out_height * out_width);
    matrix_init(&pool_output, samples, filters * 1 * 1);
    matrix_init(&d_output, samples, filters * out_height * out_width);
    matrix_init(&d_input, samples, channels * height * width);
    matrix_init(&d_output_pool, samples, filters * 1 * 1);
    layer_conv2d_init(&c, channels, height, width, filters, kernel_size, stride, &input, &output, &d_output, &d_input);
    layer_conv2d_init_values(&c, WI_ONES, BI_ONES);
    layer_maxpool2d_init(&m, filters, out_height, out_width, 3, 3, &output, &pool_output, &d_output_pool, &d_output);

    for (int i = 0; i < input.size; i++) {
        input.buffer[i] = random_normal(0.0, 1.0);
    }
    for (int i = 0; i < d_output_pool.size; i++) {
        d_output_pool.buffer[i] = 1.0;
    }

    layer_conv2d_forward(&c);
    layer_maxpool2d_forward(&m);
    layer_maxpool2d_backward(&m);
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

    // for (int filter = 0; filter < filters; filter++) {
    //     for (int channel = 0; channel < channels; channel++) {
    //         for (int i = 0; i < kernel_size; i++) {
    //             for (int j = 0; j < kernel_size; j++) {
    //                 printf("%f ", c.d_weights.buffer[filter * channels * kernel_size * kernel_size + channel * kernel_size * kernel_size + i * kernel_size + j]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //         }
    //     printf("\n");
    // }

    // for (int filter = 0; filter < filters; filter++) {
    //     printf("%f ", c.d_biases.buffer[filter]);
    // }
    // for (int sample = 0; sample < samples; sample++) {
    //     // Iterate over each filter.
    //     for (int channel = 0; channel < filters; channel++) {
    //         for (int i = 0; i < 1; i++) {
    //             for (int j = 0; j < 1; j++) {
    //                 printf("%f ", pool_output.buffer[sample * (filters * 1 * 1) + channel * (1 * 1) + i * 1 + j]);
    //             }
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    printf("output maxpool");
    for (int sample = 0; sample < samples; sample++) {
        // Iterate over each filter.
        for (int channel = 0; channel < filters; channel++) {
            for (int i = 0; i < out_height; i++) {
                for (int j = 0; j < out_width; j++) {
                    printf("%f ", output.buffer[sample * (filters * out_height * out_width) + channel * (out_height * out_width) + i * out_width + j]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
    
    printf("d_inputs maxpool");
    for (int sample = 0; sample < samples; sample++) {
        // Iterate over each filter.
        for (int channel = 0; channel < filters; channel++) {
            for (int i = 0; i < out_height; i++) {
                for (int j = 0; j < out_width; j++) {
                    printf("%f ", d_output.buffer[sample * (filters * out_height * out_width) + channel * (out_height * out_width) + i * out_width + j]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }

    printf(LAST_ERROR);
    fflush(stdout);

    layer_conv2d_free(&c);
    matrix_free(&input);
    matrix_free(&output);
    matrix_free(&pool_output);
    matrix_free(&d_output);
    matrix_free(&d_output_pool);
    matrix_free(&d_input);
}
