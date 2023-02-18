// quadratic.h
// Quadratic layer.

#ifndef QUADRATIC_H
#define QUADRATIC_H

#include "matrix.h"
#include "declspec.h"
#include "dense.h"

extern char *LAST_ERROR;

// The fully-connected quadratic layer. The layer has three trainable
// parameters: weights, biases, and quadratic weights. The forward pass is 
// calculated as X^2*Q + W*X + b. Here, X^2 represents the element-wise square
// of X. The backward pass for Q is X^2 and the backward pass for X is 
// 2*Q*X + W.
struct layer_quadratic {
    // The input and output size.
    int input_size, output_size;

    // The input and output matrices.
    struct matrix *input, *output;
    
    // The quadratic weights and biases.
    struct matrix quad, weights, biases;

    // Gradients on the outputs, inputs, weights, and biases, respectively.
    struct matrix *d_outputs, *d_inputs, d_quad, d_weights, d_biases;
};

// Initialize an empty layer object.
extern TOM_API int layer_quadratic_init(struct layer_quadratic *obj, int input_size, 
                      int output_size, struct matrix *input, 
                      struct matrix *output, struct matrix *d_outputs, 
                      struct matrix *d_inputs);

// Initialize the weights, biases, and quadratic weights.
extern TOM_API int layer_quadratic_init_values(struct layer_quadratic *obj, enum weight_initializer wi_type, enum bias_initializer bi_type, enum weight_initializer qi_type);

// Free the matrices owned by the layer.
extern TOM_API void layer_quadratic_free(struct layer_quadratic *obj);

// Perform a forward pass on the layer.
extern TOM_API void layer_quadratic_forward(struct layer_quadratic *obj);

// Perform a backward pass on the layer.
extern TOM_API void layer_quadratic_backward(struct layer_quadratic *obj);

#endif
