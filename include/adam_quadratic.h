// adam_quadratic.h
// The Adam (adaptive moment estimation) optimizer for quadratic layers.

#ifndef ADAM_QUADRATIC_H
#define ADAM_QUADRATIC_H

#include "matrix.h"
#include "quadratic.h"
#include "declspec.h"

// The Adam optimizer algorithm.
struct optimizer_adam_quadratic {
    // The quadratic layer to optimize.
    struct layer_quadratic *layer;

    // The optimizer parameters.
    double learning_rate, beta_1, beta_2, decay, epsilon;

    // The momentum matrices.
    struct matrix weight_m, bias_m, quad_m, weight_c, bias_c, quad_c;
};

// Initialize an empty Adam optimizer object.
extern TOM_API int optimizer_adam_quadratic_init(struct optimizer_adam_quadratic *obj, 
                               struct layer_quadratic *layer,
                               double learning_rate, double beta_1, 
                               double beta_2, double decay, double epsilon);

// Free the matrices owned by the optimizer.
extern TOM_API void optimizer_adam_quadratic_free(struct optimizer_adam_quadratic *obj);

// Update the layer's weights and biases.
extern TOM_API void optimizer_adam_quadratic_update(struct optimizer_adam_quadratic *obj, int iter);

#endif