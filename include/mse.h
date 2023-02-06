// mse.h
// Mean Squared Error loss function.

#ifndef MSE_H
#define MSE_H

#include "matrix.h"
#include "declspec.h"

extern char *LAST_ERROR;

// The Mean Squared Error (MSE) loss function. The output loss value is 
// calculated as sum((input - y) ** 2). The gradient on the inputs is 
// calculated as (input - y) * 2 / n_samples. Note that a forward pass
// is not strictly required for training.
struct loss_mse {
    // The input and output size.
    int input_size, output_size;

    // The input, y, and output matrices.
    struct matrix *input, *y, *output;

    // Gradients on the inputs.
    struct matrix *d_inputs;
};

// Initialize an empty MSE loss object.
extern TOM_API int loss_mse_init(struct loss_mse *obj, int input_size, struct matrix *input,
                  struct matrix *y, struct matrix *output, 
                  struct matrix *d_inputs);

// Perform a forward pass on the loss.
extern TOM_API double loss_mse_forward(struct loss_mse *obj);

// Perform a backward pass on the loss.
extern TOM_API void loss_mse_backward(struct loss_mse *obj);

#endif