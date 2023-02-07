// rmsprop.h
// RMSProp optimizer for dense layers.

#ifndef RMSPROP_H
#define RMSPROP_H

#include "matrix.h"
#include "dense.h"
#include "declspec.h"

// The Root Mean Square Propegation optimizer algorithm for dense layers.
struct optimizer_rmsprop {
    // The dense layer to optimize.
    struct layer_dense *layer;

    // The optimizer parameters.
    double learning_rate, decay, epsilon, rho;

    // The cache matrices.
    struct matrix weight_c, bias_c;
};

// Initialize an empty RMSProp optimizer object.
extern TOM_API int optimizer_rmsprop_init(struct optimizer_rmsprop *obj, struct layer_dense *layer, 
                       double learning_rate, double decay, double epsilon, double rho);

// Free the matrices owned by the optimizer.
extern TOM_API void optimizer_rmsprop_free(struct optimizer_rmsprop *obj);

// Update the layer's weights and biases.
extern TOM_API void optimizer_rmsprop_update(struct optimizer_rmsprop *obj, int iter);

#endif
