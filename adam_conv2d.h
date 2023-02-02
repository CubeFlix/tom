// adam_conv2d.h
// The Adam (adaptive moment estimation) optimizer for conv 2D layers.

#include "matrix.h"
#include "conv2d.h"

// The Adam optimizer algorithm.
struct optimizer_adam_conv2d {
    // The conv 2D layer to optimize.
    struct layer_conv2d *layer;

    // The optimizer parameters.
    double learning_rate, beta_1, beta_2, decay, epsilon;

    // The momentum matrices.
    struct matrix weight_m, bias_m, weight_c, bias_c;
};

// Initialize an empty Adam optimizer object.
int optimizer_adam_conv2d_init(struct optimizer_adam_conv2d *obj, 
                               struct layer_conv2d *layer,
                               double learning_rate, double beta_1, 
                               double beta_2, double decay, double epsilon);

// Free the matrices owned by the optimizer.
void optimizer_adam_conv2d_free(struct optimizer_adam_conv2d *obj);

// Update the layer's weights and biases.
void optimizer_adam_conv2d_update(struct optimizer_adam_conv2d *obj, int iter);