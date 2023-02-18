// sgd_bn.c
// Stochastic Gradient Descent optimizer for batch normalization layers.

#include "sgd_bn.h"
#include "matrix.h"
#include "batch_normalization.h"

// Initialize an empty SGD optimizer object.
int optimizer_sgd_bn_init(struct optimizer_sgd_bn *obj, 
                              struct layer_normalization *layer, 
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
        if (!matrix_init(&obj->gamma_m, layer->gamma.n_rows, layer->gamma.n_cols)) {
            return 0;
        }
        if (!matrix_init(&obj->beta_m, 1, layer->beta.n_cols)) {
            return 0;
        }
    }

    return 1;
}

// Free the matrices owned by the optimizer.
void optimizer_sgd_bn_free(struct optimizer_sgd_bn *obj) {
    if (obj->momentum) {
    	matrix_free(&obj->gamma_m);
    	matrix_free(&obj->beta_m);
    }
}

// Update the layer's weights and biases.
void optimizer_sgd_bn_update(struct optimizer_sgd_bn *obj, int iter) {
    // Calculate the learning rate.
    double learning_rate = obj->learning_rate;
    if (obj->decay) {
        learning_rate = learning_rate * (1.0 / (1.0 + obj->decay * (double)iter));
    }

    if (obj->momentum) {
		if (!obj->nesterov) {
        	// Use momentum.

        	// Update the weights.
        	for (int i = 0; i < obj->gamma_m.size; i++) {
            	obj->gamma_m.buffer[i] = obj->gamma_m.buffer[i] * obj->momentum - obj->layer->d_gamma.buffer[i] * learning_rate;
            	obj->layer->gamma.buffer[i] += obj->gamma_m.buffer[i];
        	}

        	// Update the biases.
        	for (int i = 0; i < obj->beta_m.size; i++) {
            	obj->beta_m.buffer[i] = obj->beta_m.buffer[i] * obj->momentum - obj->layer->d_beta.buffer[i] * learning_rate;
            	obj->layer->beta.buffer[i] += obj->beta_m.buffer[i];
        	}
		} else {
			// Use Nesterov mometum.

        	// Update the weights.
        	for (int i = 0; i < obj->gamma_m.size; i++) {
            	obj->gamma_m.buffer[i] = obj->gamma_m.buffer[i] * obj->momentum - obj->layer->d_gamma.buffer[i] * learning_rate;
            	obj->layer->gamma.buffer[i] += obj->gamma_m.buffer[i] * obj->momentum - obj->layer->d_gamma.buffer[i] * learning_rate;
        	}

        	// Update the biases.
        	for (int i = 0; i < obj->beta_m.size; i++) {
            	obj->beta_m.buffer[i] = obj->beta_m.buffer[i] * obj->momentum - obj->layer->d_beta.buffer[i] * learning_rate;
            	obj->layer->beta.buffer[i] += obj->beta_m.buffer[i] * obj->momentum - obj->layer->d_beta.buffer[i] * learning_rate;
        	}
		}
    } else {
        // Do not use momentum.
        for (int i = 0; i < obj->layer->gamma.size; i++) {
            obj->layer->gamma.buffer[i] -= obj->layer->d_gamma.buffer[i] * learning_rate;
        }
        for (int i = 0; i < obj->layer->beta.size; i++) {
            obj->layer->beta.buffer[i] -= obj->layer->d_beta.buffer[i] * learning_rate;
        }
    }
}
