// crossentropy.h
// Cross-entropy loss function.

#include "matrix.h"

extern char *LAST_ERROR;

// The cross-entropy loss function. The forward pass is calculated as 
// -log(sum(y * input)). The backward pass is calculated as (-y / d_output) / 
// n_samples. The categorical cross-entropy (softmax + cross-entropy) backward
// pass can also be computed, which places the final gradients into d_inputs.
struct loss_crossentropy {
    // The input and output size.
    int input_size, output_size;

    // The input, y, and output matrices.
    struct matrix *input, *y, *output;

    // Gradients on the inputs.
    struct matrix *d_inputs;
};

// Initialize an empty MSE loss object.
int loss_crossentropy_init(struct loss_crossentropy *obj, int input_size, 
                           struct matrix *input, struct matrix *y, 
                           struct matrix *output, struct matrix *d_inputs);

// Perform a forward pass on the loss.
double loss_crossentropy_forward(struct loss_crossentropy *obj);

// Perform a backward pass on the loss.
void loss_crossentropy_backward(struct loss_crossentropy *obj);

// Perform a backward pass on the loss and the softmax activation.
void loss_crossentropy_backward_softmax(struct loss_crossentropy *obj);