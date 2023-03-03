// adam_bn.h
// The Adam (adaptive moment estimation) optimizer for batch normalization layers.

#ifndef ADAM_BN_H
#define ADAM_BN_H

#include "matrix.h"
#include "batch_normalization.h"
#include "declspec.h"

// The Adam optimizer algorithm.
struct optimizer_adam_bn {
    // The normalization layer to optimize.
    struct layer_normalization *layer;

    // The optimizer parameters.
    double learning_rate, beta_1, beta_2, decay, epsilon;

    // The momentum matrices.
    struct matrix gamma_m, beta_m, gamma_c, beta_c;
};

// Initialize an empty Adam optimizer object.
extern TOM_API int optimizer_adam_bn_init(struct optimizer_adam_bn *obj, 
                               struct layer_normalization *layer,
                               double learning_rate, double beta_1, 
                               double beta_2, double decay, double epsilon);

// Free the matrices owned by the optimizer.
extern TOM_API void optimizer_adam_bn_free(struct optimizer_adam_bn *obj);

// Update the layer's weights and biases.
extern TOM_API void optimizer_adam_bn_update(struct optimizer_adam_bn *obj, int iter);

#endif