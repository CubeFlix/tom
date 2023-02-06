// adam.h
// The Adam (adaptive moment estimation) optimizer.

#ifndef ADAM_H
#define ADAM_H

#include "matrix.h"
#include "dense.h"
#include "declspec.h"

// The Adam optimizer algorithm.
struct optimizer_adam {
    // The dense layer to optimize.
    struct layer_dense *layer;

    // The optimizer parameters.
    double learning_rate, beta_1, beta_2, decay, epsilon;

    // The momentum matrices.
    struct matrix weight_m, bias_m, weight_c, bias_c;
};

// Initialize an empty Adam optimizer object.
extern TOM_API int optimizer_adam_init(struct optimizer_adam *obj, struct layer_dense *layer, 
                       double learning_rate, double beta_1, double beta_2,
                       double decay, double epsilon);

// Free the matrices owned by the optimizer.
extern TOM_API void optimizer_adam_free(struct optimizer_adam *obj);

// Update the layer's weights and biases.
extern TOM_API void optimizer_adam_update(struct optimizer_adam *obj, int iter);

#endif