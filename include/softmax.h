// softmax.h
// Softmax activation function.

#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "matrix.h"
#include "declspec.h"

extern char *LAST_ERROR;

// The softmax activation function. The forward pass is calculated as
// e^x/sum(e^x). The forward pass can be calculated in a numerically unstable
// method or numerically stable method. The backward pass is calculated as 
// (diag(yhat) - yhat^T * yhat).
struct activation_softmax {
    // The input and output size.
    int input_size, output_size;

    // The input and output matrices.
    struct matrix *input, *output;

    // Gradients on the outputs and inputs.
    struct matrix *d_outputs, *d_inputs;

    // Activation's internal Jacobian matrix.
    struct matrix jacobian;
};

// Initialize an empty softmax activation object.
extern TOM_API int activation_softmax_init(struct activation_softmax *obj, int input_size, 
                         struct matrix *input, struct matrix *output, 
                         struct matrix *d_outputs, struct matrix *d_inputs);

// Free the activation's matrices.
extern TOM_API void activation_softmax_free(struct activation_softmax *obj);

// Perform a forward pass on the activation.
extern TOM_API void activation_softmax_forward(struct activation_softmax *obj);

// Perform a numerically stable forward pass on the activation.
extern TOM_API void activation_softmax_forward_stable(struct activation_softmax *obj);

// Perform a backward pass on the activation.
extern TOM_API void activation_softmax_backward(struct activation_softmax *obj);

#endif
