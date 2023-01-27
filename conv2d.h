// conv2d.h
// 2D conv layer.

#ifndef CONV2D_H
#define CONV2D_H

#include <math.h>

#include "dense.h"
#include "matrix.h"
#include "random.h"

extern char *LAST_ERROR;

// The 2D conv layer. The dimensions of the input matrix are (n_samples, 
// n_channels * input_height * input_width). The dimensions of the output 
// matrix are (n_samples, n_filters * output_height * output_width). The 
// output height is (input_height - filter_size)/stride + 1. The
// output width is (input_width - filter_size)/stride + 1. The 
// total number of filters is n_filters * n_channels.
struct layer_conv2d {
    // The input and output dimensions.
    int n_channels, input_height, input_width, output_height, output_width;

    // The hyperparameter values for the conv layer.
    int n_filters, filter_size, stride;

    // The input and output matrices.
    struct matrix *input, *output;
    
    // The weights and biases.
    struct matrix weights, biases;

    // Gradients on the outputs, inputs, weights, and biases, respectively.
    struct matrix *d_outputs, *d_inputs, d_weights, d_biases;
};

// Calculate the output dimension.
#define CALC_CONV2D_OUTPUT_DIM(dim, filter_size, stride) ((dim - filter_size) / stride + 1)

// Initialize an empty layer object.
int layer_conv2d_init(struct layer_conv2d *obj, int n_channels, 
                      int input_height, int input_width, int n_filters, 
                      int filter_size, int stride, struct matrix *input, 
                      struct matrix *output, struct matrix *d_outputs, 
                      struct matrix *d_inputs);

// Initialize the weights and biases.
int layer_conv2d_init_values(struct layer_conv2d *obj, enum weight_initializer wi_type, enum bias_initializer bi_type);

// Free the matrices owned by the layer.
void layer_conv2d_free(struct layer_conv2d *obj);

// Perform a forward pass on the layer.
void layer_conv2d_forward(struct layer_conv2d *obj);

// Perform a backward pass on the layer.
void layer_conv2d_backward(struct layer_conv2d *obj);

#endif