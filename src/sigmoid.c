// sigmoid.c
// Sigmoid activation function.

#include <math.h>

#include "sigmoid.h"
#include "matrix.h"

// Initialize an empty sigmoid activation object.
int activation_sigmoid_init(struct activation_sigmoid *obj, int input_size, 
                         struct matrix *input, struct matrix *output, 
                         struct matrix *d_outputs, struct matrix *d_inputs) {
    // Set the input and output size.
    obj->input_size = input_size;
    obj->output_size = input_size;

    // Set the matrices and assert that their sizes are correct.
    obj->input = input;
    if (!(input->n_cols == input_size)) {
        // Invalid input size.
        LAST_ERROR = "Invalid input matrix size.";
        return 0;
    }

    obj->output = output;
    if (!(output->n_cols == input_size)) {
        // Invalid output size.
        LAST_ERROR = "Invalid output matrix size.";
        return 0;
    }

    obj->d_outputs = d_outputs;
    if (!(d_outputs->n_cols == input_size)) {
        // Invalid output gradient size.
        LAST_ERROR = "Invalid d_outputs matrix size.";
        return 0;
    }

    obj->d_inputs = d_inputs;
    if (!(d_inputs->n_cols == input_size)) {
        // Invalid input gradient size.
        LAST_ERROR = "Invalid d_inputs matrix size.";
        return 0;
    }

    if (!((input->n_rows == output->n_rows) && (input->n_rows == d_outputs->n_rows) && (input->n_rows == d_inputs->n_rows))) {
        // Invalid output gradient size.
        LAST_ERROR = "Input, output, d_inputs, and d_outputs matrices must have the same number of rows/samples.";
        return 0;
    }

    return 1;
}

// Perform a forward pass on the activation.
void activation_sigmoid_forward(struct activation_sigmoid *obj) {
    // Iterate over each value.
    for (int i = 0; i < obj->input->size; i++) {
        obj->output->buffer[i] = 1.0 / (1.0 + exp(-obj->output->buffer[i]));
    }
}

// Perform a backward pass on the activation.
void activation_sigmoid_backward(struct activation_sigmoid *obj) {
    // Iterate over each value.
    for (int i = 0; i < obj->input->size; i++) {
        obj->d_inputs->buffer[i] = obj->d_outputs->buffer[i] * obj->output->buffer[i] * (1.0 - obj->output->buffer[i]);
    }
}