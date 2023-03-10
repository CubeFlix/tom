// batch_normalization.h
// Batch normalization (Ioffe & Szegedy, 2015).

#ifndef BATCH_NORMALIZATION_H
#define BATCH_NORMALIZATION_H

#include "matrix.h"
#include "declspec.h"
#include "dense.h"

extern char *LAST_ERROR;

// The batch normalization layer, with running mean and variance, along with 
// affine transformation.
struct layer_normalization {
    // The input and output dimensions.
    int input_size, output_size;
    
    // The hyperparameter values for the normalization layer.
    double epsilon, momentum;

    // Gamma and beta, the parameters to be learned.
    struct matrix gamma, beta;

    // The running mean and running variance.
    struct matrix running_mean, running_variance;
    struct matrix mean, variance;

    // The input and output matrices.
    struct matrix *input, *output;
    
    // Gradients on the outputs and inputs, respectively.
    struct matrix *d_outputs, *d_inputs;

    // Gradients on gamma and beta.
    struct matrix d_gamma, d_beta;
};

// Initialize an empty layer object.
extern TOM_API int layer_normalization_init(struct layer_normalization *obj, int input_size,
                         double epsilon, double momentum, 
                         struct matrix *input, struct matrix *output,
                         struct matrix *d_outputs, struct matrix *d_inputs);

// Free the layer's matrices.
extern TOM_API void layer_normalization_free(struct layer_normalization *obj);

// Initialize gamma and beta.
extern TOM_API int layer_normalization_init_values(struct layer_normalization *obj, enum weight_initializer gamma_type, enum bias_initializer beta_type);

// Set the layer's hyper parameters.
extern TOM_API void layer_normalization_set_values(struct layer_normalization *obj, double epsilon, double momentum);

// Perform a forward pass on the layer.
extern TOM_API void layer_normalization_forward(struct layer_normalization *obj);

// Perform a forward pass on the layer, in prediction mode.
extern TOM_API void layer_normalization_forward_predict(struct layer_normalization *obj);

// Perform a backward pass on the layer.
extern TOM_API void layer_normalization_backward(struct layer_normalization *obj);

#endif