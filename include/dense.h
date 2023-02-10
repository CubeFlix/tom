// dense.h
// Dense layer.

#ifndef DENSE_H
#define DENSE_H

#include "matrix.h"
#include "declspec.h"

extern char *LAST_ERROR;

// The standard fully-connected dense layer. The layer stores the input and
// output size, along with its weights, biases, and gradients. On a forward
// pass, the layer performs the operation X*W + b on the input matrix and 
// places its output in the output matrix. On a backward pass, the gradient
// is calculated based on the gradients of the outputs, calculated by the
// following layer. The calculated gradients are then stored for the optimizer,
// and the gradients on the inputs are then passed to the preceding layer.
// Because the inputs and outputs are shared between layers, they are 
// not initialized by the layer. Similarly, with the gradients of the outputs
// and inputs, they are passed in on initialization as pointers, and are not
// stored within the layer itself. However, the weights and biases, along with
// the gradients on the weights and biases, are initialized and fully managed 
// by the layer. The dense layer supports L1 and L2 weight and bias 
// regularization.
struct layer_dense {
    // The input and output size.
    int input_size, output_size;

    // The input and output matrices.
    struct matrix *input, *output;
    
    // The weights and biases.
    struct matrix weights, biases;

    // Gradients on the outputs, inputs, weights, and biases, respectively.
    struct matrix *d_outputs, *d_inputs, d_weights, d_biases;

    // Regularization values.
    double l1_weights, l1_biases, l2_weights, l2_biases;
};

// Dense and conv 2D layer weight initializers. 
enum weight_initializer {
    // Generates zeros (0.0).
    WI_ZEROS,

    // Generates ones (1.0).
    WI_ONES,

    // Generates numbers from a uniform distribution [-1.0, 1.0]
    WI_RANDOM_UNIFORM,

    // Generates numbers from a normal distribution, centered at 0.0 with a 
    // standard deviation of 1.0.
    WI_RANDOM_NORMAL,

    // Generates numbers using Glorot normal initialization.
    WI_GLOROT_NORMAL,

    // Generates numbers using Glorot uniform initialization.
    WI_GLOROT_UNIFORM,

    // Generates numbers using He normal initialization.
    WI_HE_NORMAL,

    // Generates numbers using He uniform initialization.
    WI_HE_UNIFORM
};

// Dense amd conv 2D layer bias initializers.
enum bias_initializer {
    BI_ZEROS,
    BI_ONES
};

// Initialize an empty layer object.
extern TOM_API int layer_dense_init(struct layer_dense *obj, int input_size, 
                      int output_size, struct matrix *input, 
                      struct matrix *output, struct matrix *d_outputs, 
                      struct matrix *d_inputs);

// Initialize the weights and biases.
extern TOM_API int layer_dense_init_values(struct layer_dense *obj, enum weight_initializer wi_type, enum bias_initializer bi_type);

// Initialize regularization.
extern TOM_API void layer_dense_init_regularization(struct layer_dense *obj, double l1_weights, double l2_weights, double l1_biases, double l2_biases);

// Free the matrices owned by the layer.
extern TOM_API void layer_dense_free(struct layer_dense *obj);

// Perform a forward pass on the layer.
extern TOM_API void layer_dense_forward(struct layer_dense *obj);

// Perform a backward pass on the layer.
extern TOM_API void layer_dense_backward(struct layer_dense *obj);

// Calculate the total regularization loss for the layer.
extern TOM_API double layer_dense_calculate_regularization(struct layer_dense *obj);

#endif
