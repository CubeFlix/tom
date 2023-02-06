// conv2d.c
// 2D conv layer.

#include <math.h>

#include "conv2d.h"
#include "dense.h"
#include "matrix.h"
#include "random.h"

// Initialize an empty layer object.
int layer_conv2d_init(struct layer_conv2d *obj, int n_channels, 
                      int input_height, int input_width, int n_filters, 
                      int filter_size, int stride, struct matrix *input, 
                      struct matrix *output, struct matrix *d_outputs, 
                      struct matrix *d_inputs) {
    // Set the input and output sizes.
    obj->n_channels = n_channels;
    obj->input_height = input_height;
    obj->input_width = input_width;
    obj->output_height = CALC_CONV2D_OUTPUT_DIM(input_height, filter_size, stride);
    obj->output_width = CALC_CONV2D_OUTPUT_DIM(input_width, filter_size, stride);

    // Set the hyperparameter values.
    obj->n_filters = n_filters;
    obj->filter_size = filter_size;
    obj->stride = stride;

    // Set the matrices and assert that their sizes are correct. 
    obj->input = input;
    if (!(input->n_cols == n_channels * input_height * input_width)) {
        // Invalid input size.
        LAST_ERROR = "Invalid input matrix size.";
        return 0;
    }
    
    obj->output = output;
    if (!(output->n_cols == n_filters * obj->output_height * obj->output_width)) {
        // Invalid output size.
        LAST_ERROR = "Invalid output matrix size.";
        return 0;
    }

    obj->d_outputs = d_outputs;
    if (!(d_outputs->n_cols == n_filters * obj->output_height * obj->output_width)) {
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

    // Initialize the weights, biases, and gradients.
    if (!matrix_init(&obj->weights, n_filters * n_channels, filter_size * filter_size)) {
        return 0;
    }
    if (!matrix_init(&obj->biases, 1, n_filters)) {
        return 0;
    }
    if (!matrix_init(&obj->d_weights, n_filters * n_channels, filter_size * filter_size)) {
        return 0;
    }
    if (!matrix_init(&obj->d_biases, 1, n_filters)) {
        return 0;
    }
    return 1;
}

// Initialize the weights and biases.
int layer_conv2d_init_values(struct layer_conv2d *obj, enum weight_initializer wi_type, enum bias_initializer bi_type) {
    // Initialize the weights.
    double val;
    switch (wi_type) {
        case WI_ZEROS:
            for (int i = 0; i < (obj->weights).size; i++) {
                (obj->weights).buffer[i] = 0.0;
            }
            break;
        case WI_ONES:
            for (int i = 0; i < (obj->weights).size; i++) {
                (obj->weights).buffer[i] = 1.0;
            }
            break;
        case WI_RANDOM_UNIFORM:
            for (int i = 0; i < (obj->weights).size; i++) {
                (obj->weights).buffer[i] = random_uniform(-1.0, 2.0);
            }
            break;
        case WI_RANDOM_NORMAL:
            for (int i = 0; i < (obj->weights).size; i++) {
                (obj->weights).buffer[i] = random_normal(0.0, 1.0);
            }
            break;
        case WI_GLOROT_UNIFORM:
            val = sqrt(6.0 / (double)(obj->input_height * obj->input_width + obj->output_height * obj->output_width));
            for (int i = 0; i < (obj->weights).size; i++) {
                (obj->weights).buffer[i] = random_uniform(-val, val * 2.0);
            }
            break;
        case WI_GLOROT_NORMAL:
            val = sqrt(2.0 / (double)(obj->input_height * obj->input_width + obj->output_height * obj->output_width));
            for (int i = 0; i < (obj->weights).size; i++) {
                (obj->weights).buffer[i] = random_normal(0.0, val);
            }
            break;
        case WI_HE_UNIFORM:
            val = sqrt(6.0 / (double)(obj->input_height * obj->input_width));
            for (int i = 0; i < (obj->weights).size; i++) {
                (obj->weights).buffer[i] = random_uniform(-val, val * 2.0);
            }
            break;
        case WI_HE_NORMAL:
            val = sqrt(2.0 / (double)(obj->input_height * obj->input_width));
            for (int i = 0; i < (obj->weights).size; i++) {
                (obj->weights).buffer[i] = random_normal(0.0, val);
            }
            break;
        default:
            LAST_ERROR = "Invalid weight initializer.";
            return 0;
    }

    // Initialize the biases.
    switch (bi_type) {
        case BI_ZEROS:
            for (int i = 0; i < (obj->biases).size; i++) {
                (obj->biases).buffer[i] = 0.0;
            }
            break;
        case BI_ONES:
            for (int i = 0; i < (obj->biases).size; i++) {
                (obj->biases).buffer[i] = 1.0;
            }
            break;
        default:
            LAST_ERROR = "Invalid bias initializer.";
            return 0;
    }

    return 1;
}

// Free the matrices owned by the layer.
void layer_conv2d_free(struct layer_conv2d *obj) {
    matrix_free(&obj->weights);
    matrix_free(&obj->biases);
    matrix_free(&obj->d_weights);
    matrix_free(&obj->d_biases);
}

// Perform a forward pass on the layer.
void layer_conv2d_forward(struct layer_conv2d *obj) {
    int input_sample_size = obj->input_height * obj->input_width * obj->n_channels;
    int output_sample_size = obj->output_height * obj->output_width * obj->n_filters;
    int input_channel_size = obj->input_height * obj->input_width;
    int output_filter_size = obj->output_height * obj->output_width;
    int filter_size_per_channel = obj->n_channels * obj->filter_size * obj->filter_size;
    int filter_size = obj->filter_size * obj->filter_size;

    double sum;

    // Iterate over each sample.
    for (int sample = 0; sample < obj->input->n_rows; sample++) {
        // Iterate over each filter.
        for (int filter = 0; filter < obj->n_filters; filter++) {
            // Iterate over each stride over the height of the sample.
            for (int stride_height = 0; stride_height < obj->output_height; stride_height++) {
                // Iterate over each stride over the width of the sample.
                for (int stride_width = 0; stride_width < obj->output_width; stride_width++) {
                    // Iterate over each channel.
                    sum = 0.0;
                    for (int channel = 0; channel < obj->n_channels; channel++) {
                        // Perform the convolution operation with the kernel.
                        for (int i = 0; i < obj->filter_size; i++) {
                            for (int j = 0; j < obj->filter_size; j++) {
                                // The input value is (sample, channel, 
                                // stride_height * stride + i, stride_width * 
                                // stride + j). The kernel value is (filter, 
                                // channel, i, j). 
                                sum += obj->input->buffer[sample * input_sample_size + channel * input_channel_size + (stride_height * obj->stride + i) * obj->input_width + (stride_width * obj->stride + j)] * 
                                       obj->weights.buffer[filter * filter_size_per_channel + channel * filter_size + i * obj->filter_size + j];
                            }
                        }
                    }
                    // Set the output value at (sample, filter, stride_height, stride_width).
                    obj->output->buffer[sample * output_sample_size + filter * output_filter_size + stride_height * obj->output_width + stride_width] = sum + obj->biases.buffer[filter];
                }
            }
        }
    }
}

// Perform a backward pass on the layer.
void layer_conv2d_backward(struct layer_conv2d *obj) {
    int output_filter_size = obj->d_outputs->n_rows * obj->output_height * obj->output_width;
    int weight_filter_size = obj->n_channels * obj->filter_size * obj->filter_size;
    int weight_channel_size = obj->filter_size * obj->filter_size;
    
    double sum, one_over_n_rows = 1.0 / (double)obj->d_outputs->n_rows;
    
    // Calculate gradients on biases.
    for (int i = 0; i < obj->n_filters; i++) {
        sum = 0.0;

        // Sum the values for each filter.
        for (int j = 0; j < output_filter_size; j++) {
            sum += obj->d_outputs->buffer[i * output_filter_size + j];
        }
        obj->d_biases.buffer[i] = sum * one_over_n_rows;
    }

    // Zero the gradients.
    for (int i = 0; i < obj->d_weights.size; i++) {
        obj->d_weights.buffer[i] = 0.0;
    }
    for (int i = 0; i < obj->d_inputs->size; i++) {
        obj->d_inputs->buffer[i] = 0.0;
    }

    // Calculate gradients on weights.
    for (int sample = 0; sample < obj->input->n_rows; sample++) {
        for (int filter = 0; filter < obj->n_filters; filter++) {
            for (int channel = 0; channel < obj->n_channels; channel++) {
                for (int i = 0; i < obj->filter_size; i++) {
                    for (int j = 0; j < obj->filter_size; j++) {
                        sum = 0.0;
                        for (int stride_height = 0; stride_height < obj->output_height; stride_height++) {
                            for (int stride_width = 0; stride_width < obj->output_width; stride_width++) {
                                // Multiply the input value (sample, channel, 
                                // stride_height * stride + i, stride_width * 
                                // stride + j) with gradient value (sample, filter, i, j).
                                sum += obj->input->buffer[sample * (obj->n_channels * obj->input_height * obj->input_width) + channel * obj->input_height * obj->input_width + (stride_height * obj->stride + i) * obj->input_width + (stride_width * obj->stride + j)] * 
                                       obj->d_outputs->buffer[sample * (obj->n_filters * obj->output_height * obj->output_height) + filter * obj->output_height * obj->output_height + stride_height * obj->output_height + stride_width];
                            }
                        }
                        obj->d_weights.buffer[filter * weight_filter_size + channel * weight_channel_size + i * obj->filter_size + j] += sum;
                    }
                }
            }
        }
    }

    // Normalize weight gradients over all samples.
    for (int i = 0; i < obj->d_weights.size; i++) {
        obj->d_weights.buffer[i] *= one_over_n_rows;
    }

    // Calculate gradients on inputs.
    for (int sample = 0; sample < obj->input->n_rows; sample++) {
        for (int stride_height = 0; stride_height < obj->output_height; stride_height++) {
            for (int stride_width = 0; stride_width < obj->output_width; stride_width++) {
                for (int kern_i = 0; kern_i < obj->filter_size; kern_i++) {
                    for (int kern_j = 0; kern_j < obj->filter_size; kern_j++) {
                        for (int channel = 0; channel < obj->n_channels; channel++) {
                            sum = 0.0;
                            for (int filter = 0; filter < obj->n_filters; filter++) {
                            
                                // Multiply the weight value (filter, channel, kern_i, kern_j)
                                // with gradient value (sample, filter, stride_height,
                                // stride_width).
                                sum += obj->weights.buffer[filter * (obj->n_channels * obj->filter_size * obj->filter_size) + channel * (obj->filter_size * obj->filter_size) + kern_i * obj->filter_size + kern_j] * 
                                       obj->d_outputs->buffer[sample * (obj->n_filters * obj->output_height * obj->output_width) + filter * (obj->output_height * obj->output_width) + stride_height * obj->output_width + stride_width];
                            }
                            // Add the gradient value to (sample, channel, stride_height * filter_size + i, stride_width * filter_size + j)
                            obj->d_inputs->buffer[sample * (obj->n_channels * obj->input_height * obj->input_width) + channel * (obj->output_height * obj->output_width) + (stride_height * obj->filter_size + kern_i) * obj->output_width + (stride_width * obj->filter_size + kern_j)] += sum;
                        }
                    }
                }
            }
        }
    }

    // Normalize input gradients over all samples.
    for (int i = 0; i < obj->d_inputs->size; i++) {
        obj->d_inputs->buffer[i] *= one_over_n_rows;
    }
}