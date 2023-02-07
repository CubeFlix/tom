// sgd.h
// Stochastic Gradient Descent optimizer for dense layers.

#ifndef SGD_H
#define SGD_H

#include <stdbool.h>

#include "matrix.h"
#include "dense.h"
#include "declspec.h"

// The Stochastic Gradient Descent optimizer algorithm, with momentum and 
// decay.
struct optimizer_sgd {
    // The dense layer to optimize.
    struct layer_dense *layer;

    // The optimizer parameters.
    double learning_rate, momentum, decay;

    // Should we use Nesterov momentum.
    bool nesterov; 

    // The momentum matrices.
    struct matrix weight_m, bias_m;
};

// Initialize an empty SGD optimizer object.
extern TOM_API int optimizer_sgd_init(struct optimizer_sgd *obj, struct layer_dense *layer, 
                       double learning_rate, double momentum, double decay, bool nesterov);

// Free the matrices owned by the optimizer.
extern TOM_API void optimizer_sgd_free(struct optimizer_sgd *obj);

// Update the layer's weights and biases.
extern TOM_API void optimizer_sgd_update(struct optimizer_sgd *obj, int iter);

#endif
