// rmsprop_conv2d.h
// RMSProp optimizer for conv 2D layers.

#ifndef RMSPROP_CONV2D_H
#define RMSPROP_CONV2D_H

#include "matrix.h"
#include "conv2d.h"
#include "declspec.h"

// The Root Mean Square Propegation optimizer algorithm for conv 2D layers.
struct optimizer_rmsprop_conv2d {
    // The conv 2D layer to optimize.
    struct layer_conv2d *layer;

    // The optimizer parameters.
    double learning_rate, decay, epsilon, rho;

    // The cache matrices.
    struct matrix weight_c, bias_c;
};

// Initialize an empty RMSProp optimizer object.
extern TOM_API int optimizer_rmsprop_conv2d_init(struct optimizer_rmsprop_conv2d *obj, struct layer_conv2d *layer, 
                       double learning_rate, double decay, double epsilon, double rho);

// Free the matrices owned by the optimizer.
extern TOM_API void optimizer_rmsprop_conv2d_free(struct optimizer_rmsprop_conv2d *obj);

// Update the layer's weights and biases.
extern TOM_API void optimizer_rmsprop_conv2d_update(struct optimizer_rmsprop_conv2d *obj, int iter);

#endif
