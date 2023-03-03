// rmsprop_bn.h
// RMSProp optimizer for batch normalization layers.

#ifndef RMSPROP_BN_H
#define RMSPROP_BN_H

#include "matrix.h"
#include "batch_normalization.h"
#include "declspec.h"

// The Root Mean Square Propagation optimizer algorithm for batch normalization layers.
struct optimizer_rmsprop_bn {
    // The normalization layer to optimize.
    struct layer_normalization *layer;

    // The optimizer parameters.
    double learning_rate, decay, epsilon, rho;

    // The cache matrices.
    struct matrix gamma_c, beta_c;
};

// Initialize an empty RMSProp optimizer object.
extern TOM_API int optimizer_rmsprop_bn_init(struct optimizer_rmsprop_bn *obj, struct layer_normalization *layer, 
                       double learning_rate, double decay, double epsilon, double rho);

// Free the matrices owned by the optimizer.
extern TOM_API void optimizer_rmsprop_bn_free(struct optimizer_rmsprop_bn *obj);

// Update the layer's weights and biases.
extern TOM_API void optimizer_rmsprop_bn_update(struct optimizer_rmsprop_bn *obj, int iter);

#endif
