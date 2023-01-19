// matrix.c
// Matrix buffer objects.

#include <stdlib.h>
#include <assert.h>

#include "matrix.h"

// Initialize an empty matrix object.
int matrix_init(struct matrix *obj, int n_rows, int n_cols) {
    // Set the values for n_rows and n_cols.
    obj->n_rows = n_rows;
    obj->n_cols = n_cols;
    obj->size = n_rows * n_cols;

    // Initialize the matrix buffer.
    obj->buffer = (double *)malloc(n_rows * n_cols * sizeof(double));
    if (obj->buffer == NULL) {
        LAST_ERROR = "Failed to allocate matrix.";
        return 0;
    }
    return 1;
}

// Free a matrix buffer.
void matrix_free(struct matrix *obj) {
    // Free the buffer.
    free(obj->buffer);
}