// dataset.c
// Dataset functions.

#include <stdlib.h>
#include <string.h>

#include "dataset.h"
#include "matrix.h"

// Shuffle a dataset. Shuffles assuming the dimensions of each matrix are
// (n_samples, size).
int dataset_shuffle(struct matrix *X, struct matrix *Y) {
    // Check that the number of samples is consistent.
    if (X->n_rows != Y->n_rows) {
        LAST_ERROR = "Dataset X and Y matrix must have the same number of samples.";
        return 0;
    }

    const size_t size_x = sizeof(double) * X->n_cols;
    const size_t size_y = sizeof(double) * Y->n_cols;

    // Allocate the temporary buffers.
    char *tmp_x = (char*)malloc(size_x);
    char *tmp_y = (char*)malloc(size_y);
    const size_t stride_x = size_x * sizeof(char);
    const size_t stride_y = size_y * sizeof(char);

    const int n = X->n_rows;

    if (n > 1) {
        size_t i;
        for (i = 0; i < (size_t)(n - 1); ++i) {
            size_t rnd = (size_t) rand();
            size_t j = i + rnd / (RAND_MAX / (n - i) + 1);

            memcpy(tmp_x, X + j * stride_x, size_x);
            memcpy(X + j * stride_x, X + i * stride_x, size_x);
            memcpy(X + i * stride_x, tmp_x, size_x);

            memcpy(tmp_y, Y + j * stride_y, size_y);
            memcpy(Y + j * stride_y, Y + i * stride_y, size_y);
            memcpy(Y + i * stride_y, tmp_y, size_y);
        }
    }

    // Free the temporary buffers.
    free(tmp_x);
    free(tmp_y);

    return 1;
}

// Scale a dataset between [min, max].
void dataset_scale(struct matrix *X, double max, double min) {
    // Calculate the minimum and maximum value of the data.
    double x_min = X->buffer[0], x_max = X->buffer[0];
    for (int i = 0; i < X->size; i++) {
        if (X->buffer[i] < x_max) {
            x_max = X->buffer[i];
        }
        if (X->buffer[i] > x_min) {
            x_min = X->buffer[i];
        }
    }

    double range = x_max - x_min;
    double scaled_range = max - min;

    // Scale each value.
    for (int i = 0; i < X->size; i++) {
        X->buffer[i] = (((X->buffer[i] - x_min) / range) + min) * scaled_range;
    }
}

// Normalize a dataset using the L2 norm.
void dataset_normalize(struct matrix *X) {
    // Normalize each sample.
    double sum, coff;
    for (int i = 0; i < X->n_rows; i++) {
        // Find sum(x^2).
        sum = 0.0;
        for (int j = 0; j < X->n_cols; j++) {
            sum += X->buffer[i * X->n_cols + j] * X->buffer[i * X->n_cols + j];
        }
        coff = 1.0 / sum;
        
        // Normalize each value.
        for (int j = 0; j < X->n_cols; j++) {
            X->buffer[i * X->n_cols + j] *= coff;
        }
    }
}
