// dropout.c
// Dropout layer.

#include "dropout.h"
#include "matrix.h"

// Initialize an empty layer object.
int layer_dropout_init(struct layer_dropout *obj, int input_size, 
                      double rate, struct matrix *input, 
                      struct matrix *output, struct matrix *d_outputs, 
                      struct matrix *d_inputs) {
    // Set the input and output sizes.
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

    // Set the rate.
    obj->rate = 1.0 - rate;

    // Initialize the .
    if (!matrix_init(&obj->mask, input->n_rows, input->n_cols)) {
        return 0;
    }
    return 1;
}

// Free the matrices owned by the layer.
void layer_dropout_free(struct layer_dropout *obj) {
    matrix_free(&obj->mask);
}

// Perform a forward pass on the layer.
void layer_dropout_forward(struct layer_dropout *obj) {
    // Generate the mask and calculate the output.
    for (int i = 0; i < obj->mask.size; i++) {
        if (random_uniform(0.0, 1.0) < obj->rate) {
            obj->mask.buffer[i] = 1.0;
            obj->output->buffer[i] = obj->input->buffer[i];
        } else {
            obj->mask.buffer[i] = 0.0;
            obj->output->buffer[i] = 0.0;
        }
    }
}

// Perform a forward pass on the layer, without applying dropout.
void layer_dropout_forward_predict(struct layer_dropout *obj) {
    for (int i = 0; i < obj->mask.size; i++) {
        obj->output->buffer[i] = obj->input->buffer[i];
    }
}

// Perform a backward pass on the layer.
void layer_dropout_backward(struct layer_dropout *obj) {
    for (int i = 0; i < obj->mask.size; i++) {
        obj->d_inputs->buffer[i] = obj->d_outputs->buffer[i] * obj->mask.buffer[i];
    }
}