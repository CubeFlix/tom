// dropout.h
// Dropout layer.

#ifndef DROPOUT_H
#define DROPOUT_H

#include "matrix.h"
#include "random.h"
#include "declspec.h"

extern char *LAST_ERROR;

// The dropout layer. 
struct layer_dropout {
    // The input and output size.
    int input_size, output_size;

    // The input and output matrices.
    struct matrix *input, *output;
    
    // The current binary mask.
    struct matrix mask;

    // Gradients on the outputs and inputs.
    struct matrix *d_outputs, *d_inputs;

    // Dropout rate.
    double rate;
};

// Initialize an empty layer object.
extern TOM_API int layer_dropout_init(struct layer_dropout *obj, int input_size, 
                      double rate, struct matrix *input, 
                      struct matrix *output, struct matrix *d_outputs, 
                      struct matrix *d_inputs);

// Set the dropout layer's rate.
extern TOM_API void layer_dropout_set_rate(struct layer_dropout *obj, double rate);

// Free the matrices owned by the layer.
extern TOM_API void layer_dropout_free(struct layer_dropout *obj);

// Perform a forward pass on the layer.
extern TOM_API void layer_dropout_forward(struct layer_dropout *obj);

// Perform a forward pass on the layer, without applying dropout.
extern TOM_API void layer_dropout_forward_predict(struct layer_dropout *obj);

// Perform a backward pass on the layer.
extern TOM_API void layer_dropout_backward(struct layer_dropout *obj);

#endif