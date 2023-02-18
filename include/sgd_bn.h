// sgd_bn.h
// Stochastic Gradient Descent optimizer for batch normalization layers.

#ifndef SGD_BN_H
#define SGD_BN_H

#include <stdbool.h>

#include "matrix.h"
#include "batch_normalization.h"
#include "declspec.h"

// The Stochastic Gradient Descent optimizer algorithm, with momentum and 
// decay.
struct optimizer_sgd_bn {
    // The batch normalization layer to optimize.
    struct layer_normalization *layer;

    // The optimizer parameters.
    double learning_rate, momentum, decay;

	// Should we use Nesterov momentum.
	bool nesterov;

    // The momentum matrices.
    struct matrix gamma_m, beta_m;
};

// Initialize an empty SGD optimizer object.
extern TOM_API int optimizer_sgd_bn_init(struct optimizer_sgd_bn *obj, 
                              struct layer_normalization *layer, 
                              double learning_rate, double momentum, 
                              double decay, bool nesterov);

// Free the matrices owned by the optimizer.
extern TOM_API void optimizer_sgd_bn_free(struct optimizer_sgd_bn *obj);

// Update the layer's weights and biases.
extern TOM_API void optimizer_sgd_bn_update(struct optimizer_sgd_bn *obj, int iter);

#endif