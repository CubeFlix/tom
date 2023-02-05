// softmax.c
// Softmax activation function.

#include <math.h>
#include <stdio.h>

#include "softmax.h"
#include "matrix.h"

// Initialize an empty softmax activation object.
int activation_softmax_init(struct activation_softmax *obj, int input_size, 
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
    
    // Allocate the Jacobian cache.
    if (!matrix_init(&obj->jacobian, input_size, input_size)) {
        return 0;
    }

    return 1;
}

// Free the activation's matrices.
void activation_softmax_free(struct activation_softmax *obj) {
    matrix_free(&obj->jacobian);
}

// Perform a forward pass on the activation.
void activation_softmax_forward(struct activation_softmax *obj) {
    double sum;

    // Iterate over each sample.
    for (int i = 0; i < obj->input->n_rows; i++) {
        // Iterate over each input value.
        sum = 0.0;
        for (int j = 0; j < obj->input_size; j++) {
            obj->output->buffer[i * obj->input->n_cols + j] = exp(obj->input->buffer[i * obj->input->n_cols + j]);
            sum += obj->output->buffer[i * obj->input->n_cols + j];
        }
        // Divide each output value by the total sum.
        for (int j = 0; j < obj->input_size; j++) {
            obj->output->buffer[i * obj->input->n_cols + j] /= sum;
        }
    }
}

// Perform a numerically stable forward pass on the activation.
void activation_softmax_forward_stable(struct activation_softmax *obj) {
    double sum, max_val;

    // Iterate over each sample.
    for (int i = 0; i < obj->input->n_rows; i++) {
        // Get the maximum value of the input vector.
        max_val = -INFINITY;
        for (int j = 0; j < obj->input_size; j++) {
            if (obj->input->buffer[i * obj->input->n_cols + j] > max_val) {
                max_val = obj->input->buffer[i * obj->input->n_cols + j];
            }
        }
        // Iterate over each input value.
        sum = 0.0;
        for (int j = 0; j < obj->input_size; j++) {
            obj->output->buffer[i * obj->input->n_cols + j] = exp(obj->input->buffer[i * obj->input->n_cols + j] - max_val);
            sum += obj->output->buffer[i * obj->input->n_cols + j];
        }
        // Divide each output value by the total sum.
        for (int j = 0; j < obj->input_size; j++) {
            obj->output->buffer[i * obj->input->n_cols + j] /= sum;
        }
    }
}

// Perform a backward pass on the activation.
void activation_softmax_backward(struct activation_softmax *obj) {
    double sum;

    // Iterate over each sample.
    for (int i = 0; i < obj->input->n_rows; i++) {
        // Calculate the Jacobian matrix: x * (1(j == k) - x^T).
        for (int j = 0; j < obj->input_size; j++) {
            for (int k = 0; k < obj->input_size; k++) {
                // Calculate the dot product sum.
                obj->jacobian.buffer[j * obj->input_size + k] = obj->output->buffer[i * obj->output->n_cols + j] * ((double)(j == k) - obj->output->buffer[i * obj->output->n_cols + k]);
            }
        }

        // Calculate the dot product of the Jacobian matrix and the gradients.
        for (int j = 0; j < obj->input_size; j++) {
            sum = 0.0;
            for (int k = 0; k < obj->input_size; k++) {
                sum += obj->jacobian.buffer[j * obj->input_size + k] * obj->d_outputs->buffer[i * obj->d_outputs->n_cols + k];
            }
            obj->d_inputs->buffer[i * obj->d_inputs->n_cols + j] = sum;
        }
    }
}