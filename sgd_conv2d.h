// sgd_conv2d.h
// Stochastic Gradient Descent optimizer for conv 2D layers.

#include "matrix.h"
#include "conv2d.h"

// The Stochastic Gradient Descent optimizer algorithm, with momentum and 
// decay.
struct optimizer_sgd_conv2d {
    // The conv 2D layer to optimize.
    struct layer_conv2d *layer;

    // The optimizer parameters.
    double learning_rate, momentum, decay;

    // The momentum matrices.
    struct matrix weight_m, bias_m;
};

// Initialize an empty SGD optimizer object.
int optimizer_sgd_conv2d_init(struct optimizer_sgd_conv2d *obj, 
                              struct layer_conv2d *layer, 
                              double learning_rate, double momentum, 
                              double decay);

// Free the matrices owned by the optimizer.
void optimizer_sgd_conv2d_free(struct optimizer_sgd_conv2d *obj);

// Update the layer's weights and biases.
void optimizer_sgd_conv2d_update(struct optimizer_sgd_conv2d *obj, int iter);