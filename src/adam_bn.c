// adam_bn.c
// The Adam (adaptive moment estimation) optimizer for batch normalization layers.

#include <math.h>

#include "adam_bn.h"
#include "matrix.h"
#include "batch_normalization.h"

// Initialize an empty Adam optimizer object.
int optimizer_adam_bn_init(struct optimizer_adam_bn *obj, 
                               struct layer_normalization *layer,
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
    if (!matrix_init(&obj->gamma_m, layer->gamma.n_rows, layer->gamma.n_cols)) {
        return 0;
    }
    if (!matrix_init(&obj->beta_m, 1, layer->beta.n_cols)) {
        return 0;
    }
    if (!matrix_init(&obj->gamma_c, layer->gamma.n_rows, layer->gamma.n_cols)) {
        return 0;
    }
    if (!matrix_init(&obj->beta_c, 1, layer->beta.n_cols)) {
        return 0;
    }

    // Zero the matrices.
    for (int i = 0; i < obj->gamma_m.size; i++) {
        obj->gamma_m.buffer[i] = 0.0;
        obj->gamma_c.buffer[i] = 0.0;
    }
    for (int i = 0; i < obj->beta_m.size; i++) {
        obj->beta_m.buffer[i] = 0.0;
        obj->beta_c.buffer[i] = 0.0;
    }

    return 1;
}

// Free the matrices owned by the optimizer.
void optimizer_adam_bn_free(struct optimizer_adam_bn *obj) {
    matrix_free(&obj->gamma_m);
    matrix_free(&obj->beta_m);
    matrix_free(&obj->gamma_c);
    matrix_free(&obj->beta_c);
}

// Update the layer's weights and biases.
void optimizer_adam_bn_update(struct optimizer_adam_bn *obj, int iter) {
    // Calculate the learning rate.
    double learning_rate = obj->learning_rate;
    if (obj->decay) {
        learning_rate = learning_rate * (1.0 / (1.0 + obj->decay * (double)iter));
    }

    // Store the current corrected momentum and cache values.
    double corrected_m, corrected_c;
    double bias_correction = (1.0 / (1.0 - pow(obj->beta_1, (double)(iter + 1))));

    // Update the gamma.
    for (int i = 0; i < obj->gamma_m.size; i++) {
        // Calculate the new momentum.
        obj->gamma_m.buffer[i] = obj->gamma_m.buffer[i] * obj->beta_1 + (obj->layer->d_gamma).buffer[i] * (1.0 - obj->beta_1);

        // Calculate the new cache.
        obj->gamma_c.buffer[i] = obj->gamma_c.buffer[i] * obj->beta_2 + (obj->layer->d_gamma).buffer[i] * (obj->layer->d_gamma).buffer[i] * (1.0 - obj->beta_2);

        // Calculate the corrected momentum and cache.
        corrected_m = obj->gamma_m.buffer[i] * bias_correction;
        corrected_c = obj->gamma_c.buffer[i] * bias_correction;

        // Update the gamma.
        obj->layer->gamma.buffer[i] += -learning_rate * corrected_m / (sqrt(corrected_c) + obj->epsilon);
    }

    // Update the beta.
    for (int i = 0; i < obj->beta_m.size; i++) {
        // Calculate the new momentum.
        obj->beta_m.buffer[i] = obj->beta_m.buffer[i] * obj->beta_1 + (obj->layer->d_beta).buffer[i] * (1.0 - obj->beta_1);

        // Calculate the new cache.
        obj->beta_c.buffer[i] = obj->beta_c.buffer[i] * obj->beta_2 + (obj->layer->d_beta).buffer[i] * (obj->layer->d_beta).buffer[i] * (1.0 - obj->beta_2);

        // Calculate the corrected momentum and cache.
        corrected_m = obj->beta_m.buffer[i] * bias_correction;
        corrected_c = obj->beta_c.buffer[i] * bias_correction;

        // Update the beta.
        obj->layer->beta.buffer[i] += -learning_rate * corrected_m / (sqrt(corrected_c) + obj->epsilon);
    }
}
