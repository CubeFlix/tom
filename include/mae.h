// mae.h
// Mean Absolute Error loss function.

#ifndef MAE_H
#define MAE_H

#include "matrix.h"
#include "declspec.h"

extern char *LAST_ERROR;

// The Mean Absolute Error (MAE) loss function. The output loss value is 
// calculated as sum(abs(input - y)). The gradient on the inputs is 
// calculated as sign(input - y)/ n_samples. Note that a forward pass
// is not strictly required for training.
struct loss_mae {
    // The input and output size.
    int input_size, output_size;

    // The input, y, and output matrices.
    struct matrix *input, *y, *output;

    // Gradients on the inputs.
    struct matrix *d_inputs;
};

// Initialize an empty MAE loss object.
extern TOM_API int loss_mae_init(struct loss_mae *obj, int input_size, struct matrix *input,
                  struct matrix *y, struct matrix *output, 
                  struct matrix *d_inputs);

// Perform a forward pass on the loss.
extern TOM_API double loss_mae_forward(struct loss_mae *obj);

// Perform a backward pass on the loss.
extern TOM_API void loss_mae_backward(struct loss_mae *obj);

#endif