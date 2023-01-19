// adam.c
// The Adam (adaptive moment estimation) optimizer.

#include <math.h>

#include "adam.h"
#include "matrix.h"
#include "dense.h"

// Initialize an empty Adam optimizer object.
int optimizer_adam_init(struct optimizer_adam *obj, struct layer_dense *layer, 
                       double learning_rate, double beta_1, double beta_2,
                       double decay, double epsilon) {
    // Set the layer and the optimizer's parameters.
    obj->layer = layer;
    obj->learning_rate = learning_rate;
    obj->beta_1 = beta_1;
    obj->beta_2 = beta_2;
    obj->decay = decay;
    obj->epsilon = epsilon;

    // Initialize the momentum and cache matrices.
    if (!matrix_init(&obj->weight_m, layer->input_size, layer->output_size)) {
        return 0;
    }
    if (!matrix_init(&obj->bias_m, 1, layer->output_size)) {
        return 0;
    }
    if (!matrix_init(&obj->weight_c, layer->input_size, layer->output_size)) {
        return 0;
    }
    if (!matrix_init(&obj->bias_c, 1, layer->output_size)) {
        return 0;
    }

    return 1;
}

// Free the matrices owned by the optimizer.
void optimizer_adam_free(struct optimizer_adam *obj) {
    matrix_free(&obj->weight_m);
    matrix_free(&obj->bias_m);
    matrix_free(&obj->weight_c);
    matrix_free(&obj->bias_c);
}

// Update the layer's weights and biases.
void optimizer_adam_update(struct optimizer_adam *obj, int epoch) {
    // Calculate the learning rate.
    double learning_rate = obj->learning_rate;
    if (obj->decay) {
        learning_rate = learning_rate * (1.0 / (1.0 + obj->decay * (double)epoch));
    }

    // Store the current corrected momentum and cache values.
    double corrected_m, corrected_c;
    double bias_correction = (1.0 / (1.0 - pow(obj->beta_1, (double)(epoch + 1))));

    // Update the weights.
    for (int i = 0; i < obj->weight_m.size; i++) {
        // Calculate the new momentum.
        obj->weight_m.buffer[i] = obj->weight_m.buffer[i] * obj->beta_1 + (obj->layer->d_weights).buffer[i] * (1.0 - obj->beta_1);

        // Calculate the new cache.
        obj->weight_c.buffer[i] = obj->weight_c.buffer[i] * obj->beta_2 + (obj->layer->d_weights).buffer[i] * (obj->layer->d_weights).buffer[i] * (1.0 - obj->beta_2);

        // Calculate the corrected momentum and cache.
        corrected_m = obj->weight_m.buffer[i] * bias_correction;
        corrected_c = obj->weight_c.buffer[i] * bias_correction;

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
        corrected_m = obj->bias_m.buffer[i] * bias_correction;
        corrected_c = obj->bias_c.buffer[i] * bias_correction;

        // Update the biases.
        obj->layer->biases.buffer[i] += -learning_rate * corrected_m / (sqrt(corrected_c) + obj->epsilon);
    }
}