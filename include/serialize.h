// serialize.h
// Model serialization.

#ifndef SERIALIZE_H
#define SERIALIZE_H

#include <stdio.h>

#include "model.h"
#include "matrix.h"
#include "declspec.h"

// Serialize a matrix's data to a file.
extern TOM_API void serialize_matrix(struct matrix* obj, FILE* fp);

// Deserialize a matrix's data from a file.
extern TOM_API void deserialize_matrix(struct matrix* obj, FILE* fp);

// Serialize a layer.
extern TOM_API void serialize_layer(struct layer* obj, FILE* fp);

// Serialize a layer's parameters.
extern TOM_API void serialize_layer_params(struct layer* obj, FILE* fp);

// Deserialize a layer and add it to a model.
extern TOM_API void deserialize_layer(struct model* obj, FILE* fp);

// Deserialize a layer's parameters.
extern TOM_API void deserialize_layer_params(struct layer* obj, FILE* fp);

// Serialize a model. We serialize in two passes, once for layer information,
// and again for layer parameters.
extern TOM_API void serialize_model(struct model* obj, FILE* fp);

// Deserialize a model. Again, deserialize in two passes, loading layer data,
// initializing and finalizing the model, and then loading layer parameters.
extern TOM_API int deserialize_model(struct model* obj, FILE* fp);

#endif