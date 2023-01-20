// crossentropy.c
// Cross-entropy loss function.

#include <math.h>
#include <stdio.h>

#include "crossentropy.h"
#include "matrix.h"

// Initialize an empty cross-entropy loss object.
int loss_crossentropy_init(struct loss_crossentropy *obj, int input_size, 
                           struct matrix *input, struct matrix *y, 
                           struct matrix *output, struct matrix *d_inputs) {
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
double loss_crossentropy_forward(struct loss_crossentropy *obj) {
    double sum, sum_samples = 0.0;
    
    // Calculate the forward pass, returning the average loss over all samples.
    // Iterate over each sample.
    for (int i = 0; i < obj->input->n_rows; i++) {
        sum = 0.0;
        
        // Iterate over each input value.
        for (int j = 0; j < obj->input_size; j++) {
            // Clamp the input value.
            sum += obj->y->buffer[i * obj->input->n_cols + j] * fmin(1.0-1.0e-5, fmax(obj->input->buffer[i * obj->input->n_cols + j], 1.0e-5));
        }
        obj->output->buffer[i] = -log(sum);
        sum_samples += obj->output->buffer[i];
    }
    return sum_samples / (double)obj->input->n_rows;
}

// Perform a backward pass on the loss.
void loss_crossentropy_backward(struct loss_crossentropy *obj) {
    // Iterate over each value.
    for (int i = 0; i < obj->input->size; i++) {
        obj->d_inputs->buffer[i] = -obj->y->buffer[i] / obj->input->buffer[i] / (double)obj->input->n_rows;
    }
}

// Perform a backward pass on the loss and the softmax activation.
void loss_crossentropy_backward_softmax(struct loss_crossentropy *obj) {
    // Iterate over each value.
    for (int i = 0; i < obj->input->size; i++) {
        obj->d_inputs->buffer[i] = (obj->input->buffer[i] - obj->y->buffer[i]) / (double)obj->input->n_rows;
    }
}