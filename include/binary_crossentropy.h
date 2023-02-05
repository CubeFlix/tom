// binary_crossentropy.h
// Binary cross-entropy loss function.

#include "matrix.h"

extern char *LAST_ERROR;

// The binary cross-entropy loss function. The forward pass is calculated as 
// -(y * log(input) + (1-y) * log(1-input)). The backward pass is calculated 
// as -(y / input - (1-y) / (1 - input)). The forward pass returns the average
// loss over all samples.
struct loss_binary_crossentropy {
    // The input and output size.
    int input_size, output_size;

    // The input, y, and output matrices.
    struct matrix *input, *y, *output;

    // Gradients on the inputs.
    struct matrix *d_inputs;
};

// Initialize an empty binary cross-entropy loss object.
int loss_binary_crossentropy_init(struct loss_binary_crossentropy *obj, 
                                  int input_size, struct matrix *input, 
                                  struct matrix *y, struct matrix *output, 
                                  struct matrix *d_inputs);

// Perform a forward pass on the loss.
double loss_binary_crossentropy_forward(struct loss_binary_crossentropy *obj);

// Perform a backward pass on the loss.
void loss_binary_crossentropy_backward(struct loss_binary_crossentropy *obj);