// padding2d.c
// 2D padding layer.

#include <stdbool.h>
#include <stdlib.h>

#include "padding2d.h"
#include "matrix.h"

// Initialize an empty layer object.
int layer_padding2d_init(struct layer_padding2d *obj, int n_channels, 
                         int input_height, int input_width, int padding_x,
                         int padding_y, enum padding_type type,
                         struct matrix *input, struct matrix *output, 
                         struct matrix *d_outputs, struct matrix *d_inputs) {
    // Set the input and output sizes.
    obj->n_channels = n_channels;
    obj->input_height = input_height;
    obj->input_width = input_width;
    obj->output_height = CALC_PADDING2D_OUTPUT_DIM(input_height, padding_y);
    obj->output_width = CALC_PADDING2D_OUTPUT_DIM(input_width, padding_x);

    // Set the hyperparameter values.
    obj->padding_x = padding_x;
    obj->padding_y = padding_y;
    obj->type = type;

    // If the hyperparameter values are invalid, fail.
    if (padding_x > obj->input_width - 1) {
        LAST_ERROR = "Invalid padding dimension (must be less than input dimension).";
        return 0;
    }
    if (padding_y > obj->input_height - 1) {
        LAST_ERROR = "Invalid padding dimension (must be less than input dimension).";
        return 0;
    }

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

    // Initialize the padding caches.
    obj->output_cache = malloc(obj->output_height * obj->output_width * sizeof(int));
    if (obj->output_cache == NULL) {
        LAST_ERROR = "Failed to allocate padding cache.";
        return 0;
    }

    obj->has_caches = false;

    return 1;
}

// Free the cache owned by the layer.
void layer_padding2d_free(struct layer_padding2d *obj) {
    free(obj->output_cache);
}

// Recalculate the caches.
int layer_padding2d_recalculate_caches(struct layer_padding2d *obj) {
    // Fill the output cache with the input matrix.
    //          ---------
    //          | | | | |
    // -----    ---------
    // |1|2|    | |1|2| |
    // ----- -> ---------
    // |3|4|    | |3|4| |
    // -----    ---------
    //          | | | | |
    //          ---------
    for (int i = 0; i < obj->input_height; i++) {
        for (int j = 0; j < obj->input_width; j++) {
            // Set the output cache value at (i + padding_y, j + padding_x) to the input
            // index, i * input_width + j.
            obj->output_cache[(i + obj->padding_y) * obj->output_width + (j + obj->padding_x)] = i * obj->input_width + j;
        }
    }
    
    switch (obj->type) {
    case PADDING_ZERO:
        break;
    case PADDING_SYMMETRIC:
    {
        // Fill in the padding on the left and right sides.
        for (int i = 0; i < obj->padding_x; i++) {
            for (int j = 0; j < obj->input_height; j++) {
                // Set the output cache value at (j + padding_y, padding_x - 1 - i). The 
                // corresponding input value should be at (j, i).
                obj->output_cache[(j + obj->padding_y) * obj->output_width + obj->padding_x - 1 - i] = j * obj->input_width + i;

                // Set the output cache value at (j + padding_y, padding_x + 
                // input_width + i). The corresponding input value should be at
                // (j, input_width - 1 - i).
                obj->output_cache[(j + obj->padding_y) * obj->output_width + obj->padding_x + obj->input_width + i] = j * obj->input_width + obj->input_width - 1 - i;
            }

            for (int j = 0; j < obj->padding_y; j++) {
                // Fill in the corners.

                // Top-left corner. Set the value at (padding_y - 1 - j, 
                // padding_x - 1 - i) to (j, i).
                obj->output_cache[(obj->padding_y - 1 - j) * obj->output_width + obj->padding_x - 1 - i] = j * obj->input_width + i;

                // Top-right corner. Set the value at (padding_y - 1 - j, 
                // padding_x + input_width + i) to (j, input_width - 1 - i).
                obj->output_cache[(obj->padding_y - 1 - j) * obj->output_width + obj->padding_x + obj->input_width + i] = j * obj->input_width + obj->input_width - 1 - i;

                // Bottom-left corner. Set the value at (padding_y + input_height + j, 
                // padding_x - 1 - i) to (input_height - 1 - j, i).
                obj->output_cache[(obj->padding_y + obj->input_height + j) * obj->output_width + obj->padding_x - 1 - i] = (obj->input_height - 1 - j) * obj->input_width + i;

                // Bottom-right corner. Set the value at (padding_y + input_height + j, 
                // padding_x + input_width + i) to (input_height - 1 - j, input_width - 1 - i).
                obj->output_cache[(obj->padding_y + obj->input_height + j) * obj->output_width + obj->padding_x + obj->input_width + i] = (obj->input_height - 1 - j) * obj->input_width + obj->input_width - 1 - i;
            }
        }

        // Fill in the padding on the top and bottom sides.
        for (int i = 0; i < obj->padding_y; i++) {
            for (int j = 0; j < obj->input_width; j++) {
                // Set the output cache value at (padding_y - 1 - i, j + padding_x). The 
                // corresponding input value should be at (i, j).
                obj->output_cache[(obj->padding_y - 1 - i) * obj->output_width + j + obj->padding_x] = i * obj->input_width + j;

                // Set the output cache value at (padding_y + 
                // input_height + i, j + padding_x). The corresponding input value should be at
                // (input_height - 1 - i, j).
                obj->output_cache[(obj->padding_y + obj->input_height + i) * obj->output_width + j + obj->padding_x] = (obj->input_height - 1 - i) * obj->input_width + j;
            }
        }
        break;
    }
    case PADDING_REFLECTION:
    {
        // Fill in the padding on the left and right sides.
        for (int i = 0; i < obj->padding_x; i++) {
            for (int j = 0; j < obj->input_height; j++) {
                // Set the output cache value at (j + padding_y, padding_x - 1 - i). The 
                // corresponding input value should be at (j, i + 1).
                obj->output_cache[(j + obj->padding_y) * obj->output_width + obj->padding_x - 1 - i] = j * obj->input_width + i + 1;

                // Set the output cache value at (j + padding_y, padding_x + 
                // input_width + i). The corresponding input value should be at
                // (j, input_width - 1 - i - 1).
                obj->output_cache[(j + obj->padding_y) * obj->output_width + obj->padding_x + obj->input_width + i] = j * obj->input_width + obj->input_width - 2 - i;
            }

            for (int j = 0; j < obj->padding_y; j++) {
                // Fill in the corners.

                // Top-left corner. Set the value at (padding_y - 1 - j, 
                // padding_x - 1 - i) to (j + 1, i + 1).
                obj->output_cache[(obj->padding_y - 1 - j) * obj->output_width + obj->padding_x - 1 - i] = (j + 1) * obj->input_width + i + 1;

                // Top-right corner. Set the value at (padding_y - 1 - j, 
                // padding_x + input_width + i) to (j + 1, input_width - 1 - i - 1).
                obj->output_cache[(obj->padding_y - 1 - j) * obj->output_width + obj->padding_x + obj->input_width + i] = (j + 1) * obj->input_width + obj->input_width - 2 - i;

                // Bottom-left corner. Set the value at (padding_y + input_height + j, 
                // padding_x - 1 - i) to (input_height - 1 - j - 1, i + 1).
                obj->output_cache[(obj->padding_y + obj->input_height + j) * obj->output_width + obj->padding_x - 1 - i] = (obj->input_height - 2 - j) * obj->input_width + i + 1;

                // Bottom-right corner. Set the value at (padding_y + input_height + j, 
                // padding_x + input_width + i) to (input_height - 1 - j - 1, input_width - 1 - i - 1).
                obj->output_cache[(obj->padding_y + obj->input_height + j) * obj->output_width + obj->padding_x + obj->input_width + i] = (obj->input_height - 2 - j) * obj->input_width + obj->input_width - 2 - i;
            }
        }

        // Fill in the padding on the top and bottom sides.
        for (int i = 0; i < obj->padding_y; i++) {
            for (int j = 0; j < obj->input_width; j++) {
                // Set the output cache value at (padding_y - 1 - i, j + padding_x). The 
                // corresponding input value should be at (i + 1, j).
                obj->output_cache[(obj->padding_y - 1 - i) * obj->output_width + j + obj->padding_x] = (i + 1) * obj->input_width + j;

                // Set the output cache value at (padding_y + 
                // input_height + i, j + padding_x). The corresponding input value should be at
                // (input_height - 1 - i - 1, j).
                obj->output_cache[(obj->padding_y + obj->input_height + i) * obj->output_width + j + obj->padding_x] = (obj->input_height - 2 - i) * obj->input_width + j;
            }
        }
        break;
    }
    default:
        LAST_ERROR = "Invalid padding type.";
        return 0;
    }

    obj->has_caches = true;

    return 1;
}

// Perform a forward pass on the layer.
int layer_padding2d_forward(struct layer_padding2d *obj) {
    if (!obj->has_caches) {
        // Recalculate caches.
        if (!layer_padding2d_recalculate_caches(obj)) {
            return 0;
        }
    }

    const int sample_size = obj->n_channels * obj->output_height * obj->output_width;
    const int channel_size = obj->output_height * obj->output_width;
    const int input_sample_size = obj->n_channels * obj->input_height * obj->input_width;
    const int input_channel_size = obj->input_height * obj->input_width;

    // Iterate over each sample.
    switch (obj->type) {
    case PADDING_SYMMETRIC:
    case PADDING_REFLECTION:
    {
        // Symmetric and reflection padding.
        for (int sample = 0; sample < obj->input->n_rows; sample++) {
            // Iterate over each channel.
            for (int channel = 0; channel < obj->n_channels; channel++) {
                // Iterate over each output value.
                for (int i = 0; i < obj->output_height; i++) {
                    for (int j = 0; j < obj->output_width; j++) {
                        // Set the output value to the input value (sample, channel, output_cache[i, j]).
                        obj->output->buffer[sample * sample_size + channel * channel_size + i * obj->output_width + j] = obj->input->buffer[sample * input_sample_size + channel * input_channel_size + obj->output_cache[i * obj->output_width + j]];
                    }
                }
            }
        }
        break;
    }
    case PADDING_ZERO:
    {
        // Zero/same padding.
        // Set all the output values to zero.
        for (int i = 0; i < obj->output->size; i++) {
            obj->output->buffer[i] = 0.0;
        }

        // Set the input values.
        for (int sample = 0; sample < obj->input->n_rows; sample++) {
            // Iterate over each channel.
            for (int channel = 0; channel < obj->n_channels; channel++) {
                // Iterate over each output value.
                for (int i = 0; i < obj->input_height; i++) {
                    for (int j = 0; j < obj->input_width; j++) {
                        // Set the output value to the input value (sample, channel, output_cache[i, j]).
                        obj->output->buffer[sample * sample_size + channel * channel_size + (i + obj->padding_y) * obj->output_width + j + obj->padding_x] = obj->input->buffer[sample * input_sample_size + channel * input_channel_size + i * obj->input_width + j];
                    }
                }
            }
        }
        break;
    }
    default:
        LAST_ERROR = "Invalid padding type.";
        return 0;
    }
    return 1;
}

// Perform a backward pass on the layer.
void layer_padding2d_backward(struct layer_padding2d *obj) {
    // Copy each output gradient value to the inputs gradient.
    for (int i = 0; i < obj->input_height; i++) {
        for (int j = 0; j < obj->input_width; j++) {
            // Set the value at (i, j) to (i + padding_y, j + padding_x)
            obj->d_inputs->buffer[i * obj->input_width + j] = obj->d_outputs->buffer[(i + obj->padding_y) * obj->output_width + j + obj->padding_x];
        }
    }
}