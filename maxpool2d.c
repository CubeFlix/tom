// maxpool2d.c
// 2D max pooling layer.

#include <math.h>
#include <stdlib.h>

#include "maxpool2d.h"
#include "matrix.h"

// Initialize an empty layer object.
int layer_maxpool2d_init(struct layer_maxpool2d *obj, int n_channels, 
                         int input_height, int input_width, int pool_size, 
                         int stride, struct matrix *input, 
                         struct matrix *output, struct matrix *d_outputs, 
                         struct matrix *d_inputs) {
    // Set the input and output sizes.
    obj->n_channels = n_channels;
    obj->input_height = input_height;
    obj->input_width = input_width;
    obj->output_height = CALC_MAXPOOL2D_OUTPUT_DIM(input_height, pool_size, stride);
    obj->output_width = CALC_MAXPOOL2D_OUTPUT_DIM(input_width, pool_size, stride);

    // Set the hyperparameter values.
    obj->pool_size = pool_size;
    obj->stride = stride;

    // Set the matrices and assert that their sizes are correct. 
    obj->input = input;
    if (!(input->n_cols == n_channels * input_height * input_width)) {
        // Invalid input size.
        LAST_ERROR = "Invalid input matrix size.";
        return 0;
    }
    
    obj->output = output;
    if (!(output->n_cols == n_channels * obj->output_height * obj->output_width)) {
        // Invalid output size.
        LAST_ERROR = "Invalid output matrix size.";
        return 0;
    }

    obj->d_outputs = d_outputs;
    if (!(d_outputs->n_cols == n_channels * obj->output_height * obj->output_width)) {
        // Invalid output gradient size.
        LAST_ERROR = "Invalid d_outputs matrix size.";
        return 0;
    }

    obj->d_inputs = d_inputs;
    if (!(d_inputs->n_cols == n_channels * input_height * input_width)) {
        // Invalid input gradient size.
        LAST_ERROR = "Invalid d_inputs matrix size.";
        return 0;
    }

    if (!((input->n_rows == output->n_rows) && (input->n_rows == d_outputs->n_rows) && (input->n_rows == d_inputs->n_rows))) {
        // Invalid output gradient size.
        LAST_ERROR = "Input, output, d_inputs, and d_outputs matrices must have the same number of rows/samples.";
        return 0;
    }

    // Initialize the max pool cache. 
    obj->cache = malloc(output->n_rows * obj->n_channels * obj->output_height * obj->output_width * 2 * sizeof(int));
    if (obj->cache == NULL) {
        LAST_ERROR = "Failed to allocate max pool cache.";
        return 0;
    }
    
    return 1;
}

// Free the cache owned by the layer.
void layer_maxpool2d_free(struct layer_maxpool2d *obj) {
    free(obj->cache);
}

// Perform a forward pass on the layer.
void layer_maxpool2d_forward(struct layer_maxpool2d *obj) {
    double max, current;
    int maxX = 0, maxY = 0;
    int cache_idx;

    // Iterate over each output value.
    for (int sample = 0; sample < obj->input->n_rows; sample++) {
        // Iterate over each channel.
        for (int channel = 0; channel < obj->n_channels; channel++) {
            for (int i = 0; i < obj->output_height; i++) {
                for (int j = 0; j < obj->output_width; j++) {
                    max = -INFINITY;
                    // Perform the max pooling on the stride.
                    for (int x = 0; x < obj->pool_size; x++) {
                        for (int y = 0; y < obj->pool_size; y++) {
                            // The current value to check is (sample, channel, 
                            // i * stride + x, j * stride + y)
                            current = obj->input->buffer[sample * (obj->n_channels * obj->input_height * obj->input_width) + channel * (obj->input_height * obj->input_width) + (i * obj->stride + x) * obj->input_width + (j * obj->stride + y)];
                            if (current > max) {
                                max = current;
                                maxX = x;
                                maxY = y;
                            }
                        }
                    }
                    // Set the max value.
                    obj->output->buffer[sample * (obj->n_channels * obj->output_height * obj->output_width) + channel * (obj->output_height * obj->output_width) + i * obj->output_width + j] = max;
                
                    // Set the position of the max value in the max pool cache.
                    cache_idx = sample * (obj->n_channels * obj->output_height * obj->output_width * 2) + channel * (obj->output_height * obj->output_width * 2) + i * obj->output_width * 2 + j * 2;
                    obj->cache[cache_idx] = maxX;
                    obj->cache[cache_idx + 1] = maxY; 
                }
            }
        }
    }
}

// Perform a backward pass on the layer.
void layer_maxpool2d_backward(struct layer_maxpool2d *obj) {
    int maxX = 0, maxY = 0;
    int cache_idx;
 
    // Zero the gradients.
    for (int i = 0; i < obj->d_inputs->size; i++) {
        obj->d_inputs->buffer[i] = 0.0;
    }

    // Iterate over each sample.
    for (int sample = 0; sample < obj->input->n_rows; sample++) {
        // Iterate over each channel.
        for (int channel = 0; channel < obj->n_channels; channel++) {
            // Iterate over each output value (stride).
            for (int i = 0; i < obj->output_height; i++) {
                for (int j = 0; j < obj->output_width; j++) {
                    // Increment the max value by the gradient. The gradient 
                    // should be zero where the value is not the maximum, while
                    // the gradient should be the gradient on the input where 
                    // the value is the maximum. We will set the value at 
                    // (sample, channel, i * stride + maxX, j * stride + maxY).
                    cache_idx = sample * (obj->n_channels * obj->output_height * obj->output_width * 2) + channel * (obj->output_height * obj->output_width * 2) + i * obj->output_width * 2 + j * 2;
                    maxX = obj->cache[cache_idx];
                    maxY = obj->cache[cache_idx + 1];

                    obj->d_inputs->buffer[sample * (obj->n_channels * obj->input_height * obj->input_width) + channel * (obj->input_height * obj->input_width) + (i * obj->stride + maxX) * obj->input_width + (j * obj->stride + maxY)] += 
                        obj->d_outputs->buffer[sample * (obj->n_channels * obj->output_height * obj->output_width) + channel * (obj->output_height * obj->output_width) + i * obj->output_width + j];
                }
            }
        }
    }
}