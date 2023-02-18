// sgd_quadratic.c
// Stochastic Gradient Descent optimizer for quadratic layers.

#include "sgd_quadratic.h"
#include "matrix.h"
#include "quadratic.h"

// Initialize an empty SGD optimizer object.
int optimizer_sgd_quadratic_init(struct optimizer_sgd_quadratic *obj, 
                              struct layer_quadratic *layer, 
                              double learning_rate, double momentum, 
                              double decay, bool nesterov) {
    // Set the layer and the optimizer's parameters.
    obj->layer = layer;
    obj->learning_rate = learning_rate;
    obj->momentum = momentum;
    obj->decay = decay;
	obj->nesterov = nesterov;

    // Initialize the momentum matrices.
    if (momentum) {
        if (!matrix_init(&obj->weight_m, layer->weights.n_rows, layer->weights.n_cols)) {
            return 0;
        }
        if (!matrix_init(&obj->bias_m, 1, layer->biases.n_cols)) {
            return 0;
        }
        if (!matrix_init(&obj->quad_m, layer->weights.n_rows, layer->weights.n_cols)) {
            return 0;
        }
    }

    return 1;
}

// Free the matrices owned by the optimizer.
void optimizer_sgd_quadratic_free(struct optimizer_sgd_quadratic *obj) {
    if (obj->momentum) {
    	matrix_free(&obj->weight_m);
    	matrix_free(&obj->bias_m);
        matrix_free(&obj->quad_m);
    }
}

// Update the layer's weights and biases.
void optimizer_sgd_quadratic_update(struct optimizer_sgd_quadratic *obj, int iter) {
    // Calculate the learning rate.
    double learning_rate = obj->learning_rate;
    if (obj->decay) {
        learning_rate = learning_rate * (1.0 / (1.0 + obj->decay * (double)iter));
    }

    if (obj->momentum) {
		if (!obj->nesterov) {
        	// Use momentum.

        	// Update the weights.
        	for (int i = 0; i < obj->weight_m.size; i++) {
            	obj->weight_m.buffer[i] = obj->weight_m.buffer[i] * obj->momentum - obj->layer->d_weights.buffer[i] * learning_rate;
            	obj->layer->weights.buffer[i] += obj->weight_m.buffer[i];
        	}

        	// Update the biases.
        	for (int i = 0; i < obj->bias_m.size; i++) {
            	obj->bias_m.buffer[i] = obj->bias_m.buffer[i] * obj->momentum - obj->layer->d_biases.buffer[i] * learning_rate;
            	obj->layer->biases.buffer[i] += obj->bias_m.buffer[i];
        	}

            // Update the quads.
        	for (int i = 0; i < obj->quad_m.size; i++) {
            	obj->quad_m.buffer[i] = obj->quad_m.buffer[i] * obj->momentum - obj->layer->d_quad.buffer[i] * learning_rate;
            	obj->layer->quad.buffer[i] += obj->quad_m.buffer[i];
        	}
		} else {
			// Use Nesterov mometum.

        	// Update the weights.
        	for (int i = 0; i < obj->weight_m.size; i++) {
            	obj->weight_m.buffer[i] = obj->weight_m.buffer[i] * obj->momentum - obj->layer->d_weights.buffer[i] * learning_rate;
            	obj->layer->weights.buffer[i] += obj->weight_m.buffer[i] * obj->momentum - obj->layer->d_weights.buffer[i] * learning_rate;
        	}

        	// Update the biases.
        	for (int i = 0; i < obj->bias_m.size; i++) {
            	obj->bias_m.buffer[i] = obj->bias_m.buffer[i] * obj->momentum - obj->layer->d_biases.buffer[i] * learning_rate;
            	obj->layer->biases.buffer[i] += obj->bias_m.buffer[i] * obj->momentum - obj->layer->d_biases.buffer[i] * learning_rate;
        	}

            // Update the quads.
        	for (int i = 0; i < obj->quad_m.size; i++) {
            	obj->quad_m.buffer[i] = obj->quad_m.buffer[i] * obj->momentum - obj->layer->d_quad.buffer[i] * learning_rate;
            	obj->layer->quad.buffer[i] += obj->quad_m.buffer[i] * obj->momentum - obj->layer->d_quad.buffer[i] * learning_rate;
        	}
		}
    } else {
        // Do not use momentum.
        for (int i = 0; i < obj->layer->weights.size; i++) {
            obj->layer->weights.buffer[i] -= obj->layer->d_weights.buffer[i] * learning_rate;
        }
        for (int i = 0; i < obj->layer->biases.size; i++) {
            obj->layer->biases.buffer[i] -= obj->layer->d_biases.buffer[i] * learning_rate;
        }
        for (int i = 0; i < obj->layer->quad.size; i++) {
            obj->layer->quad.buffer[i] -= obj->layer->d_quad.buffer[i] * learning_rate;
        }
    }
}
