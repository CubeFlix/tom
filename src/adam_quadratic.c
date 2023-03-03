// adam_quadratic.c
// The Adam (adaptive moment estimation) optimizer for quadratic layers.

#include <math.h>

#include "adam_quadratic.h"
#include "matrix.h"
#include "quadratic.h"

// Initialize an empty Adam optimizer object.
int optimizer_adam_quadratic_init(struct optimizer_adam_quadratic *obj, 
                               struct layer_quadratic *layer,
                               double learning_rate, double beta_1, 
                               double beta_2, double decay, double epsilon) {
    // Set the layer and the optimizer's parameters.
    obj->layer = layer;
    obj->learning_rate = learning_rate;
    obj->beta_1 = beta_1;
    obj->beta_2 = beta_2;
    obj->decay = decay;
    obj->epsilon = epsilon;

    // Initialize the momentum and cache matrices.
    if (!matrix_init(&obj->weight_m, layer->weights.n_rows, layer->weights.n_cols)) {
        return 0;
    }
    if (!matrix_init(&obj->bias_m, 1, layer->biases.n_cols)) {
        return 0;
    }
    if (!matrix_init(&obj->quad_m, layer->quad.n_rows, layer->quad.n_cols)) {
        return 0;
    }
    if (!matrix_init(&obj->weight_c, layer->weights.n_rows, layer->weights.n_cols)) {
        return 0;
    }
    if (!matrix_init(&obj->bias_c, 1, layer->biases.n_cols)) {
        return 0;
    }
    if (!matrix_init(&obj->quad_c, layer->quad.n_rows, layer->quad.n_cols)) {
        return 0;
    }

    for (int i = 0; i < obj->weight_m.size; i++) {
        obj->weight_m.buffer[i] = 0.0;
        obj->weight_c.buffer[i] = 0.0;
    }
    for (int i = 0; i < obj->bias_m.size; i++) {
        obj->bias_m.buffer[i] = 0.0;
        obj->bias_c.buffer[i] = 0.0;
    }
    for (int i = 0; i < obj->quad_m.size; i++) {
        obj->quad_m.buffer[i] = 0.0;
        obj->quad_c.buffer[i] = 0.0;
    }

    return 1;
}

// Free the matrices owned by the optimizer.
void optimizer_adam_quadratic_free(struct optimizer_adam_quadratic *obj) {
    matrix_free(&obj->weight_m);
    matrix_free(&obj->bias_m);
    matrix_free(&obj->quad_m);
    matrix_free(&obj->weight_c);
    matrix_free(&obj->bias_c);
    matrix_free(&obj->quad_c);
}

// Update the layer's weights and biases.
void optimizer_adam_quadratic_update(struct optimizer_adam_quadratic *obj, int iter) {
    // Calculate the learning rate.
    double learning_rate = obj->learning_rate;
    if (obj->decay) {
        learning_rate = learning_rate * (1.0 / (1.0 + obj->decay * (double)iter));
    }

    // Store the current corrected momentum and cache values.
    double corrected_m, corrected_c;
    double bias_correction_m = (1.0 / (1.0 - pow(obj->beta_1, (double)(iter + 1))));
    double bias_correction_c = (1.0 / (1.0 - pow(obj->beta_2, (double)(iter + 1))));

    // Update the weights.
    for (int i = 0; i < obj->weight_m.size; i++) {
        // Calculate the new momentum.
        obj->weight_m.buffer[i] = obj->weight_m.buffer[i] * obj->beta_1 + (obj->layer->d_weights).buffer[i] * (1.0 - obj->beta_1);

        // Calculate the new cache.
        obj->weight_c.buffer[i] = obj->weight_c.buffer[i] * obj->beta_2 + (obj->layer->d_weights).buffer[i] * (obj->layer->d_weights).buffer[i] * (1.0 - obj->beta_2);

        // Calculate the corrected momentum and cache.
        corrected_m = obj->weight_m.buffer[i] * bias_correction_m;
        corrected_c = obj->weight_c.buffer[i] * bias_correction_c;

        // Update the weights.
        obj->layer->weights.buffer[i] += -learning_rate * corrected_m / (sqrt(corrected_c) + obj->epsilon);
    }

    // Update the biases.
    for (int i = 0; i < obj->bias_m.size; i++) {
        // Calculate the new momentum.
        obj->bias_m.buffer[i] = obj->bias_m.buffer[i] * obj->beta_1 + (obj->layer->d_biases).buffer[i] * (1.0 - obj->beta_1);

        // Calculate the new cache.
        obj->bias_c.buffer[i] = obj->bias_c.buffer[i] * obj->beta_2 + (obj->layer->d_biases).buffer[i] * (obj->layer->d_biases).buffer[i] * (1.0 - obj->beta_2);

        // Calculate the corrected momentum and cache.
        corrected_m = obj->bias_m.buffer[i] * bias_correction_m;
        corrected_c = obj->bias_c.buffer[i] * bias_correction_c;

        // Update the biases.
        obj->layer->biases.buffer[i] += -learning_rate * corrected_m / (sqrt(corrected_c) + obj->epsilon);
    }

    // Update the quads.
    for (int i = 0; i < obj->quad_m.size; i++) {
        // Calculate the new momentum.
        obj->quad_m.buffer[i] = obj->quad_m.buffer[i] * obj->beta_1 + (obj->layer->d_quad).buffer[i] * (1.0 - obj->beta_1);

        // Calculate the new cache.
        obj->quad_c.buffer[i] = obj->quad_c.buffer[i] * obj->beta_2 + (obj->layer->d_quad).buffer[i] * (obj->layer->d_quad).buffer[i] * (1.0 - obj->beta_2);

        // Calculate the corrected momentum and cache.
        corrected_m = obj->quad_m.buffer[i] * bias_correction_m;
        corrected_c = obj->quad_c.buffer[i] * bias_correction_c;

        // Update the quads.
        obj->layer->quad.buffer[i] += -learning_rate * corrected_m / (sqrt(corrected_c) + obj->epsilon);
    }
}
