// sigmoid.h
// Sigmoid activation function.

#include "matrix.h"

extern char *LAST_ERROR;

// The sigmoid activation function. The forward pass is calculated as
// 1 / (1 + e^(-x)). The backward pass is calculated as d_output * x * (1-x).
struct activation_sigmoid {
    // The input and output size.
    int input_size, output_size;

    // The input and output matrices.
    struct matrix *input, *output;

    // Gradients on the outputs and inputs.
    struct matrix *d_outputs, *d_inputs;
};

// Initialize an empty sigmoid activation object.
int activation_sigmoid_init(struct activation_sigmoid *obj, int input_size, 
                         struct matrix *input, struct matrix *output, 
                         struct matrix *d_outputs, struct matrix *d_inputs);

// Perform a forward pass on the activation.
void activation_sigmoid_forward(struct activation_sigmoid *obj);

// Perform a backward pass on the activation.
void activation_sigmoid_backward(struct activation_sigmoid *obj);