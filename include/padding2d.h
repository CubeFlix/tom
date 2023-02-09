// padding2d.h
// 2D padding layer.

#ifndef PADDING2D_H
#define PADDING2D_H

#include <stdbool.h>

#include "matrix.h"
#include "declspec.h"

extern char *LAST_ERROR;

// Padding type enum.
enum padding_type {
    // Zero padding, or "same padding".
    PADDING_ZERO,

    // Symmetric padding.
    PADDING_SYMMETRIC,

    // Reflection padding.
    PADDING_REFLECTION
};

// The 2D padding layer.
struct layer_padding2d {
    // The input and output dimensions.
    int n_channels, input_height, input_width, output_height, output_width;
    
    // The hyperparameter values for the padding layer. The padding is applied
    // twice to each dimension, on both sides.
    int padding_x, padding_y;
    enum padding_type type;

    // The input and output matrices.
    struct matrix *input, *output;

    // Padding output cache. For each output value, it stores the index of the 
    // corresponding input value. The cache only needs to be calculated once,
    // on the first forward pass.
    int *output_cache;

    // Padding gradient cache. For each input value, the value is 1.0 
    // plus 1.0 multiplied by the number of times said input value appears in
    // the output padding. For example, if a particular input value is repeated
    // twice in the output padding, its corresponding gradient cache will be 
    // 3.0. For zero-padding, all values will be 1.0. The cache only needs
    // to be calculated once, on the first forward pass.
    struct matrix grad_cache;

    bool has_caches;
    
    // Gradients on the outputs and inputs, respectively.
    struct matrix *d_outputs, *d_inputs;
};

// Calculate the output dimension.
#define CALC_PADDING2D_OUTPUT_DIM(dim, padding) (dim + padding * 2)

// Initialize an empty layer object.
extern TOM_API int layer_padding2d_init(struct layer_padding2d *obj, int n_channels,
                         int input_height, int input_width, int padding_x,
                         int padding_y, enum padding_type type,
                         struct matrix *input, struct matrix *output,
                         struct matrix *d_outputs, struct matrix *d_inputs);

// Free the cache owned by the layer.
extern TOM_API void layer_padding2d_free(struct layer_padding2d *obj);

// Set the padding type.
extern TOM_API void layer_padding2d_set_type(struct layer_padding2d *obj, enum padding_type type);

// Recalculate the caches.
extern TOM_API int layer_padding2d_recalculate_caches(struct layer_padding2d *obj);

// Perform a forward pass on the layer.
extern TOM_API int layer_padding2d_forward(struct layer_padding2d *obj);

// Perform a backward pass on the layer.
extern TOM_API void layer_padding2d_backward(struct layer_padding2d *obj);

#endif