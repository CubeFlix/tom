// rmsprop_conv2d.c
// RMSProp optimizer for conv 2D layers.

#include <math.h>

#include "rmsprop_conv2d.h"
#include "matrix.h"
#include "conv2d.h"
#include "declspec.h"

// Initialize an empty RMSProp optimizer object.
int optimizer_rmsprop_conv2d_init(struct optimizer_rmsprop_conv2d *obj, struct layer_conv2d *layer, 
                       double learning_rate, double decay, double epsilon, double rho) {
	// Set the layer and the optimizer's parameters.
	obj->layer = layer;
	obj->learning_rate = learning_rate;
	obj->decay = decay;
	obj->epsilon = epsilon;
	obj->rho = rho;

	// Initialize the cache matrices.
	if (!matrix_init(&obj->weight_c, layer->weights.n_rows, layer->weights.n_cols)) {
		return 0;
	}
	if (!matrix_init(&obj->bias_c, 1, layer->biases.n_cols)) {
		return 0;
	}

	// Zero the matrices.
    for (int i = 0; i < obj->weight_c.size; i++) {
        obj->weight_c.buffer[i] = 0.0;
    }
    for (int i = 0; i < obj->bias_c.size; i++) {
        obj->bias_c.buffer[i] = 0.0;
    }

	return 1;
}

// Free the matrices owned by the optimizer.
void optimizer_rmsprop_conv2d_free(struct optimizer_rmsprop_conv2d *obj) {
	matrix_free(&obj->weight_c);
	matrix_free(&obj->bias_c);
}

// Update the layer's weights and biases.
void optimizer_rmsprop_conv2d_update(struct optimizer_rmsprop_conv2d *obj, int iter) {
	// Calculate the learning rate.
	double learning_rate = obj->learning_rate;
	if (obj->decay) {
		learning_rate = learning_rate * (1.0 / (1.0 + obj->decay * (double)iter));
	}

	// Update the weight cache and weights.
	for (int i = 0; i < obj->weight_c.size; i++) {
		obj->weight_c.buffer[i] = obj->rho * obj->weight_c.buffer[i] + (1.0 - obj->rho) * obj->layer->d_weights.buffer[i] * obj->layer->d_weights.buffer[i];
		obj->layer->weights.buffer[i] += -learning_rate * obj->layer->d_weights.buffer[i] / sqrt(obj->weight_c.buffer[i] + obj->epsilon);
	}
	for (int i = 0; i < obj->bias_c.size; i++) {
		obj->bias_c.buffer[i] = obj->rho * obj->bias_c.buffer[i] + (1.0 - obj->rho) * obj->layer->d_biases.buffer[i] * obj->layer->d_biases.buffer[i];
		obj->layer->biases.buffer[i] += -learning_rate * obj->layer->d_biases.buffer[i] / sqrt(obj->bias_c.buffer[i] + obj->epsilon);
	}
}
