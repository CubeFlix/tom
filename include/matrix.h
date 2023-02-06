// matrix.h
// Matrix buffer objects.

#ifndef MATRIX_H
#define MATRIX_H

#include "declspec.h"

extern char *LAST_ERROR;

// Matrix struct. We store the matrix data as a buffer of doubles, row by row.
// The location of the item at (row, col) is (row * n_cols + col) * 
// sizeof(double).
struct matrix {
    int n_rows, n_cols, size;
    double *buffer;
};

// Initialize an empty matrix object.
extern TOM_API int matrix_init(struct matrix *obj, int n_rows, int n_cols);

// Free a matrix buffer.
extern TOM_API void matrix_free(struct matrix *obj);

#endif