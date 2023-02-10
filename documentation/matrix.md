# Matrix

`tom` uses the `matrix` data structure to represent matrices. The `matrix` structure stores its dimensions, along with a pointer to a dynamically-allocated buffer of double-precision floating-point values.

```
struct matrix {
    int n_rows, n_cols, size;
    double *buffer;
};
```

- `n_rows`: Number of rows in the matrix.
- `n_cols`: Number of columns in the matrix.
- `size`: Calculated as `n_rows * n_cols`.
- `buffer`: A dynamically-allocated buffer of type `double`, with size `size`.

Matrices can be allocated with the `matrix_init` function, and freed with `matrix_free`. See below for an example:

```
struct matrix m;
int n_rows = 5, n_cols = 3;

int main(void) {
    // Allocate the matrix buffer.
    QUIT_ON_ERROR(matrix_init(&m, n_rows, n_cols));

    // Do stuff with the matrix.
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            m.buffer[i * n_cols + j] = i + j;
        }
    }

    // Free the matrix buffer.
    matrix_free(&m);

    return 0;
}
```


## `matrix`

Matrix struct. We store the matrix data as a buffer of doubles, row by row. The location of the item at (`row`, `col`) is `(row * n_cols + col) * sizeof(double)`.

```
struct matrix {
    int n_rows, n_cols, size;
    double *buffer;
};
```

### `int matrix_init(struct matrix *obj, int n_rows, int n_cols)`

Initialize an empty matrix object. `obj` should be a pointer to the matrix to initialize. `n_rows` and `n_cols` determine the number of rows and columns of the new matrix, respectively. `matrix_init` sets the `n_rows`, `n_cols`, `size`, and `buffer` fields of the matrix. Returns `1` if successful, otherwise it returns `0`.

### `void matrix_free(struct matrix *obj)`

Free a matrix buffer. `obj` should be a pointer to the matrix to free. If successful, `matrix_free` should free the buffer and set the `buffer` field to `NULL`.