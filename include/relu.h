// relu.h
// Rectified Linear Unit activation function.

#ifndef RELU_H
#define RELU_H

#include "matrix.h"
#include "declspec.h"

extern char *LAST_ERROR;

// The Rectified Linear Unit (RELU) activation layer. The output is calculated
// as 1 * x if x > 0, or 0 if x <= 0. The gradient is calculated as d_output
// if x > 0, or 0 if x <= 0.
struct activation_relu {
    // The input and output size.
    int input_size, output_size;

    // The input and output matrices.
    struct matrix *input, *output;

    // Gradients on the outputs and inputs.
    struct matrix *d_outputs, *d_inputs;
};

// Initialize an empty RELU activation object.
extern TOM_API int activation_relu_init(struct activation_relu *obj, int input_size, 
                         struct matrix *input, struct matrix *output, 
                         struct matrix *d_outputs, struct matrix *d_inputs);

// Perform a forward pass on the activation.
extern TOM_API void activation_relu_forward(struct activation_relu *obj);

// Perform a backward pass on the activation.
extern TOM_API void activation_relu_backward(struct activation_relu *obj);

#endif