// rmsprop_quadratic.h
// RMSProp optimizer for quadratic layers.

#ifndef RMSPROP_QUADRATIC_H
#define RMSPROP_QUADRATIC_H

#include "matrix.h"
#include "quadratic.h"
#include "declspec.h"

// The Root Mean Square Propagation optimizer algorithm for quadratic layers.
struct optimizer_rmsprop_quadratic {
    // The quadratic layer to optimize.
    struct layer_quadratic *layer;

    // The optimizer parameters.
    double learning_rate, decay, epsilon, rho;

    // The cache matrices.
    struct matrix weight_c, bias_c, quad_c;
};

// Initialize an empty RMSProp optimizer object.
extern TOM_API int optimizer_rmsprop_quadratic_init(struct optimizer_rmsprop_quadratic *obj, struct layer_quadratic *layer, 
                       double learning_rate, double decay, double epsilon, double rho);

// Free the matrices owned by the optimizer.
extern TOM_API void optimizer_rmsprop_quadratic_free(struct optimizer_rmsprop_quadratic *obj);

// Update the layer's weights and biases.
extern TOM_API void optimizer_rmsprop_quadratic_update(struct optimizer_rmsprop_quadratic *obj, int iter);

#endif
