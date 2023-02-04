// serialize.h
// Model serialization.

#include <stdlib.h>

#include "model.h"
#include "matrix.h"

// Serialize a matrix's data to a file.
void serialize_matrix(struct matrix* obj, FILE* fp);

// Deserialize a matrix's data from a file.
void deserialize_matrix(struct matrix* obj, FILE* fp);