// binary_crossentropy.c
// Binary cross-entropy loss function.

#include "binary_crossentropy.h"
#include "matrix.h"

// Initialize an empty binary cross-entropy loss object.
int loss_binary_crossentropy_init(struct loss_binary_crossentropy *obj, 
                                  int input_size, struct matrix *input, 
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
double loss_binary_crossentropy_forward(struct loss_binary_crossentropy *obj) {
    double clipped, sum, sum_samples = 0.0;
    double one_over_input_size = 1.0 / (double)obj->input_size;
    
    // Calculate the forward pass, returning the average loss over all samples.
    // Iterate over each sample.
    for (int i = 0; i < obj->input->n_rows; i++) {
        sum = 0.0;
        
        // Iterate over each input value.
        for (int j = 0; j < obj->input_size; j++) {
            clipped = fmin(1.0-1.0e-5, fmax(obj->input->buffer[i * obj->input_size + j], 1.0e-5));
            if (obj->y->buffer[i * obj->input_size + j] != 0.0) {
                sum += -log(clipped);
            } else {
                sum += -log(1.0 - clipped);
            }
        }
        obj->output->buffer[i] = sum * one_over_input_size;
        sum_samples += obj->output->buffer[i];
    }
    return sum_samples / (double)obj->input->n_rows;
}

// Perform a backward pass on the loss.
void loss_binary_crossentropy_backward(struct loss_binary_crossentropy *obj) {
    double clipped;
    double one_over_input_rows = 1.0 / (double)obj->input->n_rows;

    // Iterate over each item.
    for (int i = 0; i < obj->input->size; i++) {
        clipped = fmin(1.0-1.0e-5, fmax(obj->input->buffer[i], 1.0e-5));
        obj->d_inputs->buffer[i] = -(obj->y->buffer[i] / clipped - (1.0 - obj->y->buffer[i]) / (1.0 - clipped)) * one_over_input_rows;
    }
}