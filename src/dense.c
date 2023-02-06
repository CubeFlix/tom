// dense.c
// Dense layer.

#include <math.h>

#include "dense.h"
#include "matrix.h"
#include "random.h"

// Initialize an empty layer object.
int layer_dense_init(struct layer_dense *obj, int input_size, 
                      int output_size, struct matrix *input, 
                      struct matrix *output, struct matrix *d_outputs, 
                      struct matrix *d_inputs) {
    // Set the input and output sizes.
    obj->input_size = input_size;
    obj->output_size = output_size;

    // Set the matrices and assert that their sizes are correct. 
    obj->input = input;
    if (!(input->n_cols == input_size)) {
        // Invalid input size.
        LAST_ERROR = "Invalid input matrix size.";
        return 0;
    }
    
    obj->output = output;
    if (!(output->n_cols == output_size)) {
        // Invalid output size.
        LAST_ERROR = "Invalid output matrix size.";
        return 0;
    }

    obj->d_outputs = d_outputs;
    if (!(d_outputs->n_cols == output_size)) {
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

    // Initialize the weights, biases, and gradients.
    if (!matrix_init(&obj->weights, input_size, output_size)) {
        return 0;
    }
    if (!matrix_init(&obj->biases, 1, output_size)) {
        return 0;
    }
    if (!matrix_init(&obj->d_weights, input_size, output_size)) {
        return 0;
    }
    if (!matrix_init(&obj->d_biases, 1, output_size)) {
        return 0;
    }
    return 1;
}

// Initialize the weights and biases.
int layer_dense_init_values(struct layer_dense *obj, enum weight_initializer wi_type, enum bias_initializer bi_type) {
    // Initialize the weights.
    double val;
    switch (wi_type) {
        case WI_ZEROS:
            for (int i = 0; i < (obj->weights).size; i++) {
                (obj->weights).buffer[i] = 0.0;
            }
            break;
        case WI_ONES:
            for (int i = 0; i < (obj->weights).size; i++) {
                (obj->weights).buffer[i] = 1.0;
            }
            break;
        case WI_RANDOM_UNIFORM:
            for (int i = 0; i < (obj->weights).size; i++) {
                (obj->weights).buffer[i] = random_uniform(-1.0, 2.0);
            }
            break;
        case WI_RANDOM_NORMAL:
            for (int i = 0; i < (obj->weights).size; i++) {
                (obj->weights).buffer[i] = random_normal(0.0, 1.0);
            }
            break;
        case WI_GLOROT_UNIFORM:
            val = sqrt(6.0 / (double)(obj->input_size + obj->output_size));
            for (int i = 0; i < (obj->weights).size; i++) {
                (obj->weights).buffer[i] = random_uniform(-val, val * 2.0);
            }
            break;
        case WI_GLOROT_NORMAL:
            val = sqrt(2.0 / (double)(obj->input_size + obj->output_size));
            for (int i = 0; i < (obj->weights).size; i++) {
                (obj->weights).buffer[i] = random_normal(0.0, val);
            }
            break;
        case WI_HE_UNIFORM:
            val = sqrt(6.0 / (double)(obj->input_size));
            for (int i = 0; i < (obj->weights).size; i++) {
                (obj->weights).buffer[i] = random_uniform(-val, val * 2.0);
            }
            break;
        case WI_HE_NORMAL:
            val = sqrt(2.0 / (double)(obj->input_size));
            for (int i = 0; i < (obj->weights).size; i++) {
                (obj->weights).buffer[i] = random_normal(0.0, val);
            }
            break;
        default:
            LAST_ERROR = "Invalid weight initializer.";
            return 0;
    }

    // Initialize the biases.
    switch (bi_type) {
        case BI_ZEROS:
            for (int i = 0; i < (obj->biases).size; i++) {
                (obj->biases).buffer[i] = 0.0;
            }
            break;
        case BI_ONES:
            for (int i = 0; i < (obj->biases).size; i++) {
                (obj->biases).buffer[i] = 1.0;
            }
            break;
        default:
            LAST_ERROR = "Invalid bias initializer.";
            return 0;
    }

    return 1;
}

// Initialize regularization.
void layer_dense_init_regularization(struct layer_dense *obj, double l1_weights, double l2_weights, double l1_biases, double l2_biases) {
    obj->l1_weights = l1_weights;
    obj->l2_weights = l2_weights;
    obj->l1_biases = l1_biases;
    obj->l2_biases = l2_biases;
}

// Free the matrices owned by the layer.
void layer_dense_free(struct layer_dense *obj) {
    matrix_free(&obj->weights);
    matrix_free(&obj->biases);
    matrix_free(&obj->d_weights);
    matrix_free(&obj->d_biases);
}

// Perform a forward pass on the layer.
void layer_dense_forward(struct layer_dense *obj) {
    // Calculate W*x + b.
    int n_samples = obj->input->n_rows;
    int input_size = obj->input_size;
    int output_size = obj->output_size;
    double sum;

    // Iterate over each sample.
    for (int i = 0; i < n_samples; i++) {
        // Iterate over each output value.
        for (int j = 0; j < output_size; j++) {
            // Calculate the output value.
            sum = 0.0;

            // Calculate the dot product of the inputs and weights, and add the biases.
            for (int k = 0; k < input_size; k++) {
                sum += obj->input->buffer[i * input_size + k] * obj->weights.buffer[k * output_size + j];
            }
            obj->output->buffer[i * output_size + j] = sum + obj->biases.buffer[j];
        }
    }
}

// Perform a backward pass on the layer.
void layer_dense_backward(struct layer_dense *obj) {
    int n_samples = obj->input->n_rows;
    int input_size = obj->input_size;
    int output_size = obj->output_size;
    double sum;

    // Calculate d_weights = x^T*d_outputs.
    // Iterate over each input value.
    for (int i = 0; i < input_size; i++) {
        // Iterate over each output value.
        for (int j = 0; j < output_size; j++) {
            // Calculate the gradient on the weights.
            sum = 0.0;

            // Calculate the dot product of the transform on the inputs and 
            // the gradients from the following layer.
            for (int k = 0; k < n_samples; k++) {
                sum += obj->input->buffer[k * input_size + i] * obj->d_outputs->buffer[k * output_size + j];
            }
            obj->d_weights.buffer[i * output_size + j] = sum;
        }
    }

    // Calculate weight regularization.
    if (obj->l1_weights) {
        for (int i = 0; i < obj->d_weights.size; i++) {
            obj->d_weights.buffer[i] += obj->l1_weights * copysign(1.0, obj->weights.buffer[i]);
        }
    }
    if (obj->l2_weights) {
        for (int i = 0; i < obj->d_weights.size; i++) {
            obj->d_weights.buffer[i] += obj->l2_weights * obj->weights.buffer[i] * 2.0;
        }
    }

    // Calculate d_biases = sum(d_outputs).
    for (int i = 0; i < output_size; i++) {
        // Calculate the sum of the column.
        sum = 0.0;
        for (int j = 0; j < n_samples; j++) {
            sum += obj->d_outputs->buffer[j * output_size + i];
        }
        obj->d_biases.buffer[i] = sum;
    }

    // Calculate bias regularization.
    if (obj->l1_biases) {
        for (int i = 0; i < obj->d_biases.size; i++) {
            obj->d_biases.buffer[i] += obj->l1_biases * copysign(1.0, obj->biases.buffer[i]);
        }
    }
    if (obj->l2_biases) {
        for (int i = 0; i < obj->d_biases.size; i++) {
            obj->d_biases.buffer[i] += obj->l2_biases * obj->biases.buffer[i] * 2.0;
        }
    }

    // Calculate d_inputs = d_outputs * W^T.
    // Iterate over each sample.
    for (int i = 0; i < n_samples; i++) {
        // Iterate over each input value.
        for (int j = 0; j < input_size; j++) {
            // Calculate the gradient on the inputs.
            sum = 0.0;

            // Calculate the dot product of the gradients from the following
            // layer and the transform on the weights.
            for (int k = 0; k < output_size; k++) {
                sum += obj->d_outputs->buffer[i * output_size + k] * obj->weights.buffer[j * output_size + k];
            }
            obj->d_inputs->buffer[i * input_size + j] = sum;
        }
    }
}

// Calculate the total regularization loss for the layer.
double layer_dense_calculate_regularization(struct layer_dense *obj) {
    double reg_loss = 0.0;

    // Iterate over each weight and bias value.
    for (int i = 0; i < obj->weights.size; i++) {
        reg_loss += (fabs(obj->weights.buffer[i]) * obj->l1_weights) + (obj->weights.buffer[i] * obj->weights.buffer[i] * obj->l2_weights);
    }
    for (int i = 0; i < obj->biases.size; i++) {
        reg_loss += (fabs(obj->biases.buffer[i]) * obj->l1_biases) + (obj->biases.buffer[i] * obj->biases.buffer[i] * obj->l2_biases);
    }

    return reg_loss;
}