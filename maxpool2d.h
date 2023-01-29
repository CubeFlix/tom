// maxpool2d.h
// 2D max pooling layer.

#ifndef MAXPOOL2D_H
#define MAXPOOL2D_H

#include "matrix.h"

extern char *LAST_ERROR;

// The 2D max pooling layer. 
struct layer_maxpool2d {
    // The input and output dimensions.
    int n_channels, input_height, input_width, output_height, output_width;

    // The hyperparameter values for the pooling layer.
    int pool_size, stride;

    // The input and output matrices.
    struct matrix *input, *output;

    // Max pooling cache. We store the x and y position of the maximum value 
    // for each stride. The total dimensions of the cache are (samples, 
    // channels, output_height, output_width, 2). 
    int *cache;
    
    // Gradients on the outputs and inputs, respectively.
    struct matrix *d_outputs, *d_inputs;
};

// Calculate the output dimension.
#define CALC_MAXPOOL2D_OUTPUT_DIM(dim, pool_size, stride) ((dim - pool_size) / stride + 1)

// Initialize an empty layer object.
int layer_maxpool2d_init(struct layer_maxpool2d *obj, int n_channels, 
                         int input_height, int input_width, int pool_size, 
                         int stride, struct matrix *input, 
                         struct matrix *output, struct matrix *d_outputs, 
                         struct matrix *d_inputs);

// Free the cache owned by the layer.
void layer_maxpool2d_free(struct layer_maxpool2d *obj);

// Perform a forward pass on the layer.
void layer_maxpool2d_forward(struct layer_maxpool2d *obj);

// Perform a backward pass on the layer.
void layer_maxpool2d_backward(struct layer_maxpool2d *obj);

#endif