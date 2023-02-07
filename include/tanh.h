// tanh.h
// Hyperbolic tangent activation function.

#ifndef TANH_H
#define TANH_H

#include "matrix.h"
#include "declspec.h"

extern char *LAST_ERROR;

// The hyperbolic tangent (tanh) activation function. The forward pass is 
// calculated as (e^x - e^(-x))/(e^x + e^(-x)). The backward pass is calculated
// as d_output * (1 - y^2).
struct activation_tanh {
    // The input and output size.
    int input_size, output_size;

    // The input and output matrices.
    struct matrix *input, *output;

    // Gradients on the outputs and inputs.
    struct matrix *d_outputs, *d_inputs;
};

// Initialize an empty tanh activation object.
extern TOM_API int activation_tanh_init(struct activation_tanh *obj, int input_size, 
                         struct matrix *input, struct matrix *output, 
                         struct matrix *d_outputs, struct matrix *d_inputs);

// Perform a forward pass on the activation.
extern TOM_API void activation_tanh_forward(struct activation_tanh *obj);

// Perform a backward pass on the activation.
extern TOM_API void activation_tanh_backward(struct activation_tanh *obj);

#endif