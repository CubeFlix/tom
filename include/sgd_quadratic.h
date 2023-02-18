// sgd_quadratic.h
// Stochastic Gradient Descent optimizer for quadratic layers.

#ifndef SGD_QUADRATIC_H
#define SGD_QUADRATIC_H

#include <stdbool.h>

#include "matrix.h"
#include "quadratic.h"
#include "declspec.h"

// The Stochastic Gradient Descent optimizer algorithm, with momentum and 
// decay.
struct optimizer_sgd_quadratic {
    // The quadratic layer to optimize.
    struct layer_quadratic *layer;

    // The optimizer parameters.
    double learning_rate, momentum, decay;

	// Should we use Nesterov momentum.
	bool nesterov;

    // The momentum matrices.
    struct matrix weight_m, bias_m, quad_m;
};

// Initialize an empty SGD optimizer object.
extern TOM_API int optimizer_sgd_quadratic_init(struct optimizer_sgd_quadratic *obj, 
                              struct layer_quadratic *layer, 
                              double learning_rate, double momentum, 
                              double decay, bool nesterov);

// Free the matrices owned by the optimizer.
extern TOM_API void optimizer_sgd_quadratic_free(struct optimizer_sgd_quadratic *obj);

// Update the layer's weights and biases.
extern TOM_API void optimizer_sgd_quadratic_update(struct optimizer_sgd_quadratic *obj, int iter);

#endif