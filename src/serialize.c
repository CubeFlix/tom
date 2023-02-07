// serialize.c
// Model serialization.

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#include "serialize.h"
#include "model.h"
#include "matrix.h"
#include "dense.h"
#include "dropout.h"
#include "conv2d.h"
#include "maxpool2d.h"
#include "leaky_relu.h"

// Serialize a matrix's data.
int serialize_matrix(struct matrix* obj, FILE* fp) {
	// Serialize the matrix values.
	if (fwrite(obj->buffer, sizeof(double), obj->size, fp) != (size_t)obj->size) {
		LAST_ERROR = "Failed to write file.";
		return 0;
	}
	return 1;
}

// Deserialize a matrix's data.
int deserialize_matrix(struct matrix* obj, FILE* fp) {
	// Deserialize the matrix values.
	if (fread(obj->buffer, sizeof(double), obj->size, fp) != (size_t)obj->size) {
		LAST_ERROR = "Failed to read file.";
		return 0;
	}
	return 1;
}

// Serialize a layer.
int serialize_layer(struct layer* obj, FILE* fp) {
	// Write the layer type.
	if (fwrite(&obj->type, sizeof(enum layer_type), 1, fp) != 1) {
		LAST_ERROR = "Failed to write file.";
		return 0;
    }

	// Write the layer parameters.
	if (fwrite(&obj->input_size, sizeof(int), 10, fp) != 10) {
		LAST_ERROR = "Failed to write file.";
		return 0;
	}

	return 1;
}

// Serialize a layer's parameters.
int serialize_layer_params(struct layer* obj, FILE* fp) {
	// Write the trainable parameters.
	switch (obj->type) {
	case LAYER_DENSE:
		if (!serialize_matrix(&((struct layer_dense*)(obj->obj))->weights, fp)) {
			return 0;	
		}
		if (!serialize_matrix(&((struct layer_dense*)(obj->obj))->biases, fp)) {
			return 0;
		}
		break;
	case LAYER_CONV2D:
		if (!serialize_matrix(&((struct layer_conv2d*)(obj->obj))->weights, fp)) {
			return 0;	
		}
		if (!serialize_matrix(&((struct layer_conv2d*)(obj->obj))->biases, fp)) {
			return 0;
		}
		break;
	case LAYER_DROPOUT:
		if (fwrite(&((struct layer_dropout*)(obj->obj))->rate, sizeof(double), 1, fp) != 1) {
			LAST_ERROR = "Failed to write file.";
			return 0;
		}
		break;
	case LAYER_LEAKY_RELU:
		if (fwrite(&((struct activation_leaky_relu*)(obj->obj))->rate, sizeof(double), 1, fp) != 1) {
			LAST_ERROR = "Failed to write file.";
			return 0;
		}
		break;
	default:
		break;
	}
	
	return 1;
}

// Deserialize a layer and add it to a model.
int deserialize_layer(struct model* obj, FILE* fp) {
	// Read the layer type.
	enum layer_type ltype;
	if (fread(&ltype, sizeof(enum layer_type), 1, fp) != 1) {LAST_ERROR = "Failed to read file."; return 0;}

	// Read the layer parameters.
	int input_size, output_size;
	int input_channels, input_height, input_width;
	int output_channels, output_height, output_width;
	int filter_size, stride;
	if (fread(&input_size, sizeof(int), 1, fp) != 1) {LAST_ERROR = "Failed to read file."; return 0;}
	if (fread(&output_size, sizeof(int), 1, fp) != 1) {LAST_ERROR = "Failed to read file."; return 0;}
	if (fread(&input_channels, sizeof(int), 1, fp) != 1) {LAST_ERROR = "Failed to read file."; return 0;}
	if (fread(&input_height, sizeof(int), 1, fp) != 1) {LAST_ERROR = "Failed to read file."; return 0;}
	if (fread(&input_width, sizeof(int), 1, fp) != 1) {LAST_ERROR = "Failed to read file."; return 0;}
	if (fread(&output_channels, sizeof(int), 1, fp) != 1) {LAST_ERROR = "Failed to read file."; return 0;}
	if (fread(&output_height, sizeof(int), 1, fp) != 1) {LAST_ERROR = "Failed to read file."; return 0;}
	if (fread(&output_width, sizeof(int), 1, fp) != 1) {LAST_ERROR = "Failed to read file."; return 0;}
	if (fread(&filter_size, sizeof(int), 1, fp) != 1) {LAST_ERROR = "Failed to read file."; return 0;}
	if (fread(&stride, sizeof(int), 1, fp) != 1) {LAST_ERROR = "Failed to read file."; return 0;}

	switch (ltype) {
	case LAYER_CONV2D:
		// Conv 2D layer.
		model_add_conv2d_layer(obj, input_channels, input_height, input_width, output_channels, filter_size, stride);
		break;
	case LAYER_MAXPOOL2D:
		// Max pooling 2D layer.
		model_add_maxpool2d_layer(obj, input_channels, input_height, input_width, filter_size, stride);
		break;
	default:
		model_add_layer(obj, ltype, input_size, output_size);
		break;
	}
	
	return 1;
}

// Deserialize a layer's parameters.
int deserialize_layer_params(struct layer* obj, FILE* fp) {
	switch (obj->type) {
	case LAYER_CONV2D:
		// Conv 2D layer.
		if (!deserialize_matrix(&((struct layer_conv2d*)(obj->obj))->weights, fp)) {
			return 0;
		}
		if (!deserialize_matrix(&((struct layer_conv2d*)(obj->obj))->biases, fp)) {
			return 0;
		}
		break;
	case LAYER_DENSE:
		// Dense layer.
		if (!deserialize_matrix(&((struct layer_dense*)(obj->obj))->weights, fp)) {
			return 0;
		}
		if (!deserialize_matrix(&((struct layer_dense*)(obj->obj))->biases, fp)) {
			return 0;
		}
		break;
	case LAYER_DROPOUT:
		// Dropout layer.
		if (fread(&((struct layer_dropout*)(obj->obj))->rate, sizeof(double), 1, fp) != 1) {
			LAST_ERROR = "Failed to read file.";
			return 0;
		}
		break;
	case LAYER_LEAKY_RELU:
		// Leaky RELU layer.
		if (fread(&((struct activation_leaky_relu*)(obj->obj))->rate, sizeof(double), 1, fp) != 1) {
			LAST_ERROR = "Failed to read file.";
			return 0;
		}
		break;
	default:
		break;
	}
	return 1;
}

// Serialize a model. We serialize in two passes, once for layer information,
// and again for layer parameters.
int serialize_model(struct model* obj, FILE* fp) {
	// Write the number of layers.
	if (fwrite(&obj->n_layers, sizeof(int), 1, fp) != 1) {
		LAST_ERROR = "Failed to write file.";
		return 0;
	}

	// Write the loss type.
	if (fwrite(&obj->loss.type, sizeof(enum loss_type), 1, fp) != 1) {
		LAST_ERROR = "Failed to write file.";
		return 0;
	}

	// Write each layer.
	struct layer* current = obj->first;
	do {
		if (!serialize_layer(current, fp)) {
			return 0;
		}
		current = current->next;
	} while (current != NULL);
	
	// Write the layer parameters.
	current = obj->first;
	do {
		if (!serialize_layer_params(current, fp)) {
			return 0;
		}
		current = current->next;
	} while (current != NULL);
	
	return 1;
}

// Deserialize a model. Again, deserialize in two passes, loading layer data,
// initializing and finalizing the model, and then loading layer parameters.
int deserialize_model(struct model* obj, FILE* fp) {
	// Get the number of layers.
	int n_layers;
	if (fread(&n_layers, sizeof(int), 1, fp) != 1) {
		LAST_ERROR = "Failed to read file.";
		return 0;
	}

	// Get the loss type.
	if (fread(&obj->loss.type, sizeof(enum loss_type), 1, fp) != 1) {
		LAST_ERROR = "Failed to read file.";
		return 0;
	}

	// Load each layer.
	for (int i = 0; i < n_layers; i++) {
		if (!deserialize_layer(obj, fp)) {
			return 0;
		}
	}

	// Finalize the model.
	if (!model_finalize(obj)) {
		return 0;
	}

	// Load the layer parameters.
	struct layer* current = obj->first;
	do {
		if (!deserialize_layer_params(current, fp)) {
			return 0;
		}
		current = current->next;
	} while (current != NULL);

	return 1;
}
