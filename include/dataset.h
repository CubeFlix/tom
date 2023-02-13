// dataset.h
// Dataset functions.

#ifndef DATASET_H
#define DATASET_H

#include "matrix.h"
#include "declspec.h"

extern char *LAST_ERROR;

// Shuffle a dataset. Shuffles assuming the dimensions of each matrix are
// (n_samples, size).
extern TOM_API int dataset_shuffle(struct matrix *X, struct matrix *Y);

// Scale a dataset between [min, max].
extern TOM_API void dataset_scale(struct matrix *X, double max, double min);

// Normalize a dataset using the L2 norm.
extern TOM_API void dataset_normalize(struct matrix *X);

#endif