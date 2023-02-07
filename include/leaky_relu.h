// leaky_relu.h
// Leaky Rectified Linear Unit activation function.

#ifndef LEAKY_RELU_H
#define LEAKY_RELU_H

#include "matrix.h"
#include "declspec.h"

extern char *LAST_ERROR;

// The Leaky Rectified Linear Unit (RELU) activation layer. The output is 
// calculated as 1 * x if x > 0, or rate * x if x <= 0. The gradient is 
// calculated as d_output if x > 0, or rate * d_output if x <= 0.
struct activation_leaky_relu {
    // The input and output size.
    int input_size, output_size;
	
	// The leaky RELU rate.
	double rate;

    // The input and output matrices.
    struct matrix *input, *output;

    // Gradients on the outputs and inputs.
    struct matrix *d_outputs, *d_inputs;
};

// Initialize an empty leaky RELU activation object.
extern TOM_API int activation_leaky_relu_init(struct activation_leaky_relu *obj, int input_size, 
                         double rate, struct matrix *input, struct matrix *output, 
                         struct matrix *d_outputs, struct matrix *d_inputs);

// Perform a forward pass on the activation.
extern TOM_API void activation_leaky_relu_forward(struct activation_leaky_relu *obj);

// Perform a backward pass on the activation.
extern TOM_API void activation_leaky_relu_backward(struct activation_leaky_relu *obj);

#endif