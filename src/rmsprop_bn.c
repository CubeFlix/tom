// rmsprop_bn.c
// RMSProp optimizer for batch normalization layers.

#include <math.h>

#include "rmsprop_bn.h"
#include "matrix.h"
#include "batch_normalization.h"

// Initialize an empty RMSProp optimizer object.
int optimizer_rmsprop_bn_init(struct optimizer_rmsprop_bn *obj, struct layer_normalization *layer, 
                       double learning_rate, double decay, double epsilon, double rho) {
	// Set the layer and the optimizer's parameters.
	obj->layer = layer;
	obj->learning_rate = learning_rate;
	obj->decay = decay;
	obj->epsilon = epsilon;
	obj->rho = rho;

	// Initialize the cache matrices.
	if (!matrix_init(&obj->gamma_c, layer->gamma.n_rows, layer->gamma.n_cols)) {
		return 0;
	}
	if (!matrix_init(&obj->beta_c, 1, layer->beta.n_cols)) {
		return 0;
	}

	// Zero the matrices.
    for (int i = 0; i < obj->gamma_c.size; i++) {
        obj->gamma_c.buffer[i] = 0.0;
    }
    for (int i = 0; i < obj->beta_c.size; i++) {
        obj->beta_c.buffer[i] = 0.0;
    }

	return 1;
}

// Free the matrices owned by the optimizer.
void optimizer_rmsprop_bn_free(struct optimizer_rmsprop_bn *obj) {
	matrix_free(&obj->gamma_c);
	matrix_free(&obj->beta_c);
}

// Update the layer's weights and biases.
void optimizer_rmsprop_bn_update(struct optimizer_rmsprop_bn *obj, int iter) {
	// Calculate the learning rate.
	double learning_rate = obj->learning_rate;
	if (obj->decay) {
		learning_rate = learning_rate * (1.0 / (1.0 + obj->decay * (double)iter));
	}

	// Update the weight cache and weights.
	for (int i = 0; i < obj->gamma_c.size; i++) {
		obj->gamma_c.buffer[i] = obj->rho * obj->gamma_c.buffer[i] + (1.0 - obj->rho) * obj->layer->d_gamma.buffer[i] * obj->layer->d_gamma.buffer[i];
		obj->layer->gamma.buffer[i] += -learning_rate * obj->layer->d_gamma.buffer[i] / sqrt(obj->gamma_c.buffer[i] + obj->epsilon);
	}
	for (int i = 0; i < obj->beta_c.size; i++) {
		obj->beta_c.buffer[i] = obj->rho * obj->beta_c.buffer[i] + (1.0 - obj->rho) * obj->layer->d_beta.buffer[i] * obj->layer->d_beta.buffer[i];
		obj->layer->beta.buffer[i] += -learning_rate * obj->layer->d_beta.buffer[i] / sqrt(obj->beta_c.buffer[i] + obj->epsilon);
	}
}
