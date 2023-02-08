// mae.c
// Mean Absolute Error loss function.

#include <math.h>

#include "mae.h"
#include "matrix.h"

// Initialize an empty MAE loss object.
int loss_mae_init(struct loss_mae *obj, int input_size, struct matrix *input,
                  struct matrix *y, struct matrix *output, 
                  struct matrix *d_inputs) {
    // Set the input and output size.
    obj->input_size = input_size;
    obj->output_size = 1;

    // Set the matrices and assert that their sizes are correct.
    obj->input = input;
    if (!(input->n_cols == input_size)) {
        // Invalid input size.
        LAST_ERROR = "Invalid input matrix size.";
        return 0;
    }

    obj->y = y;
    if (!(y->n_cols == input_size)) {
        // Invalid input size.
        LAST_ERROR = "Invalid y matrix size.";
        return 0;
    }

    obj->output = output;
    if (!(output->n_cols == 1)) {
        // Invalid output size.
        LAST_ERROR = "Invalid output matrix size.";
        return 0;
    }

    obj->d_inputs = d_inputs;
    if (!(d_inputs->n_cols == input_size)) {
        // Invalid input gradient size.
        LAST_ERROR = "Invalid d_inputs matrix size.";
        return 0;
    }

    if (!((input->n_rows == output->n_rows) && (input->n_rows == y->n_rows) && (input->n_rows == d_inputs->n_rows))) {
        // Invalid output gradient size.
        LAST_ERROR = "Input, output, y, and d_inputs matrices must have the same number of rows/samples.";
        return 0;
    }

    return 1;
}

// Perform a forward pass on the loss.
double loss_mae_forward(struct loss_mae *obj) {
    // Calculate the forward pass, returning the average loss over all samples.
    // Iterate over each sample.
    double sum, sum_samples;
    sum_samples = 0.0;
    for (int i = 0; i < obj->input->n_rows; i++) {
        sum = 0.0;

        // Iterate over each input value.
        for (int j = 0; j < obj->input->n_cols; j++) {
            sum += fabs(obj->input->buffer[i * obj->input->n_cols + j] - obj->y->buffer[i * obj->input->n_cols + j]);
        }
        obj->output->buffer[i] = sum;
        sum_samples += sum;
    }
    return sum_samples / (double)obj->input->n_rows;
}

// Perform a backward pass on the loss.
void loss_mae_backward(struct loss_mae *obj) {
    double coff = 1.0 / (double)obj->input_size;
    
    // Iterate over each value.
    for (int i = 0; i < obj->input->size; i++) {
        obj->d_inputs->buffer[i] = copysign(coff, obj->input->buffer[i] - obj->y->buffer[i]);
    }
}