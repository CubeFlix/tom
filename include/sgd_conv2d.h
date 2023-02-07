// sgd_conv2d.h
// Stochastic Gradient Descent optimizer for conv 2D layers.

#ifndef SGD_CONV2D_H
#define SGD_CONV2D_H

#include <stdbool.h>

#include "matrix.h"
#include "conv2d.h"
#include "declspec.h"

// The Stochastic Gradient Descent optimizer algorithm, with momentum and 
// decay.
struct optimizer_sgd_conv2d {
    // The conv 2D layer to optimize.
    struct layer_conv2d *layer;

    // The optimizer parameters.
    double learning_rate, momentum, decay;

	// Should we use Nesterov momentum.
	bool nesterov;

    // The momentum matrices.
    struct matrix weight_m, bias_m;
};

// Initialize an empty SGD optimizer object.
extern TOM_API int optimizer_sgd_conv2d_init(struct optimizer_sgd_conv2d *obj, 
                              struct layer_conv2d *layer, 
                              double learning_rate, double momentum, 
                              double decay, bool nesterov);

// Free the matrices owned by the optimizer.
extern TOM_API void optimizer_sgd_conv2d_free(struct optimizer_sgd_conv2d *obj);

// Update the layer's weights and biases.
extern TOM_API void optimizer_sgd_conv2d_update(struct optimizer_sgd_conv2d *obj, int iter);

#endif