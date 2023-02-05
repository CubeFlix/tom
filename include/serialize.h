// serialize.h
// Model serialization.

#include <stdlib.h>

#include "model.h"
#include "matrix.h"

// Serialize a matrix's data to a file.
void serialize_matrix(struct matrix* obj, FILE* fp);

// Deserialize a matrix's data from a file.
void deserialize_matrix(struct matrix* obj, FILE* fp);

// Serialize a layer.
void serialize_layer(struct layer* obj, FILE* fp);

// Serialize a layer's parameters.
void serialize_layer_params(struct layer* obj, FILE* fp);

// Deserialize a layer and add it to a model.
void deserialize_layer(struct model* obj, FILE* fp);

// Deserialize a layer's parameters.
void deserialize_layer_params(struct layer* obj, FILE* fp);

// Serialize a model. We serialize in two passes, once for layer information,
// and again for layer parameters.
void serialize_model(struct model* obj, FILE* fp);

// Deserialize a model. Again, deserialize in two passes, loading layer data,
// initializing and finalizing the model, and then loading layer parameters.
int deserialize_model(struct model* obj, FILE* fp);