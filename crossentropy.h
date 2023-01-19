// crossentropy.h
// Categorical cross-entropy loss function.

#include "matrix.h"

extern char *LAST_ERROR;

// The categorical cross-entropy loss function.
struct loss_mse {
    // The input and output size.
    int input_size, output_size;

    // The input, y, and output matrices.
    struct matrix *input, *y, *output;

    // Gradients on the inputs.
    struct matrix *d_inputs;
};

// Initialize an empty MSE loss object.
int loss_mse_init(struct loss_mse *obj, int input_size, struct matrix *input,
                  struct matrix *y, struct matrix *output, 
                  struct matrix *d_inputs);

// Perform a forward pass on the loss.
double loss_mse_forward(struct loss_mse *obj);

// Perform a backward pass on the activation.
void loss_mse_backward(struct loss_mse *obj);