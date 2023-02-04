// serialize.c
// Model serialization.

#include <stdio.h>
#include <stdbool.h>

#include "serialize.h"
#include "model.h"
#include "matrix.h"

// Serialize a matrix's data.
void serialize_matrix(struct matrix* obj, FILE* fp) {
	// Serialize the matrix values.
	fwrite(obj->buffer, sizeof(double), obj->size, fp);
}

// Deserialize a matrix's data.
void deserialize_matrix(struct matrix* obj, FILE* fp) {
	// Deserialize the matrix values.
	fread(obj->buffer, sizeof(double), obj->size, fp);
}

// Serialize a layer.
void serialize_layer(struct layer* obj, FILE* fp) {
	// Write the layer type.
	fwrite(&obj->type, sizeof(enum), 1, fp);

	// Write the layer parameters.
	fwrite(&obj->input_size, sizeof(int), 10, fp);

	// Write the trainable parameters.
	switch (obj->type) {
	case LAYER_DENSE:
		serialize_matrix((struct layer_dense*)(obj->obj)->weights, fp);
		serialize_matrix((struct layer_dense*)(obj->obj)->biases, fp);
		break;
	case LAYER_CONV2D:
		serialize_matrix((struct layer_dense*)(obj->obj)->weights, fp);
		serialize_matrix((struct layer_dense*)(obj->obj)->biases, fp);
		break;
	case LAYER_DROPOUT:
		fwrite(&(struct layer_dropout*)(obj->obj)->rate, sizeof(double), 1, fp);
		break;
	}
}

// Deserialize a layer and add it to a model.
void deserialize_layer(struct model* obj, FILE* fp) {
	// Read the layer type.
	enum layer_type ltype;
	fread(&ltype, sizeof(enum), 1, fp);

	// Read the layer parameters.
	int input_size, output_size;
	int input_channels, input_height, input_width;
	int output_channels, output_height, output_width;
	int filter_size, stride;
	fread(&input_size, sizeof(int), 10, fp);

	struct layer* l;

	switch (ltype) {
	case LAYER_CONV2D:
		// Conv 2D layer.
		l = model_add_conv2d_layer(obj, input_channels, input_height, input_width, output_channels, filter_size, stride);
		deserialize_matrix((struct layer_conv2d*)(l->obj)->weights, fp);
		deserialize_matrix((struct layer_conv2d*)(l->obj)->biases, fp);
		break;
	case LAYER_MAXPOOL2D:
		// Max pooling 2D layer.
		l = model_add_maxpool2d_layer(obj, input_channels, input_height, input_width, filter_size, stride);
		break;
	case LAYER_DENSE:
		// Dense layer.
		l = model_add_layer(obj, ltype, input_size, output_size);
		deserialize_matrix((struct layer_dense*)(l->obj)->weights, fp);
		deserialize_matrix((struct layer_dense*)(l->obj)->biases, fp);
		break;
	case LAYER_DROPOUT:
		// Dropout layer.
		l = model_add_layer(obj, ltype, input_type, output_type);
		fread(&(struct layer_dropout*)(l->obj)->rate, sizeof(double), 1, fp);
		break;
	default:
		l = model_add_layer(obj, ltype, input_type, output_type);
		break;
	}
}

// Serialize a model.
void serialize_model(struct model* obj, FILE* fp) {
	// Write the number of layers.
	fwrite(&obj->n_layers, sizeof(int), 1, fp);

	// Write the loss type.
	fwrite(&obj->loss.type, sizeof(enum), 1, fp);

	// Write each layer.
	struct layer* current = obj->first;
	do {
		serialize_layer(current, fp);
		current = current->next;
	} while (current != NULL);
}

// Deserialize a model.
int deserialize_model(struct model* obj, FILE* fp) {
	// Get the number of layers.
	int n_layers;
	fread(&n_layers, sizeof(int), 1, fp);

	// Get the loss type.
	fread(&obj->loss.type, sizeof(enum), 1, fp);

	// Load each layer.
	for (int i = 0; i < n_layers; i++) {
		deserialize_layer(obj, fp);
	}

	// Finalize the model.
	return model_finalize(obj);
}