// batch_normalization.c
// Batch normalization (Ioffe & Szegedy, 2015).

#include <math.h>

#include "batch_normalization.h"
#include "matrix.h"
#include "random.h"

// Initialize an empty layer object.
int layer_normalization_init(struct layer_normalization *obj, int input_size,
                         double epsilon, double momentum,
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

    obj->epsilon = epsilon;
    obj->momentum = momentum;

    // Allocate the running mean and running variance.
    if (!matrix_init(&obj->running_mean, 1, input_size)) {
        return 0;
    }
    if (!matrix_init(&obj->running_variance, 1, input_size)) {
        return 0;
    }
    if (!matrix_init(&obj->mean, 1, input_size)) {
        return 0;
    }
    if (!matrix_init(&obj->variance, 1, input_size)) {
        return 0;
    }
    if (!matrix_init(&obj->gamma, 1, input_size)) {
        return 0;
    }
    if (!matrix_init(&obj->beta, 1, input_size)) {
        return 0;
    }
    if (!matrix_init(&obj->d_gamma, 1, input_size)) {
        return 0;
    }
    if (!matrix_init(&obj->d_beta, 1, input_size)) {
        return 0;
    }

    // Initialize the running mean and running variance.
    for (int i = 0; i < obj->input_size; i++) {
        obj->running_mean.buffer[i] = 0.0;
        obj->running_variance.buffer[i] = 0.0;
    }

    return 1;
}


// Free the layer's matrices.
void layer_normalization_free(struct layer_normalization *obj) {
    matrix_free(&obj->running_mean);
    matrix_free(&obj->running_variance);
    matrix_free(&obj->mean);
    matrix_free(&obj->variance);
    matrix_free(&obj->gamma);
    matrix_free(&obj->beta);
    matrix_free(&obj->d_gamma);
    matrix_free(&obj->d_beta);
}

// Initialize gamma and beta.
int layer_normalization_init_values(struct layer_normalization *obj, enum weight_initializer gamma_type, enum bias_initializer beta_type) {
    // Initialize gamma.
    double val;
    switch (gamma_type) {
        case WI_ZEROS:
            for (int i = 0; i < (obj->gamma).size; i++) {
                (obj->gamma).buffer[i] = 0.0;
            }
            break;
        case WI_ONES:
            for (int i = 0; i < (obj->gamma).size; i++) {
                (obj->gamma).buffer[i] = 1.0;
            }
            break;
        case WI_RANDOM_UNIFORM:
            for (int i = 0; i < (obj->gamma).size; i++) {
                (obj->gamma).buffer[i] = random_uniform(-1.0, 2.0);
            }
            break;
        case WI_RANDOM_NORMAL:
            for (int i = 0; i < (obj->gamma).size; i++) {
                (obj->gamma).buffer[i] = random_normal(0.0, 1.0);
            }
            break;
        case WI_GLOROT_UNIFORM:
            val = sqrt(6.0 / (double)(obj->input_size + obj->output_size));
            for (int i = 0; i < (obj->gamma).size; i++) {
                (obj->gamma).buffer[i] = random_uniform(-val, val * 2.0);
            }
            break;
        case WI_GLOROT_NORMAL:
            val = sqrt(2.0 / (double)(obj->input_size + obj->output_size));
            for (int i = 0; i < (obj->gamma).size; i++) {
                (obj->gamma).buffer[i] = random_normal(0.0, val);
            }
            break;
        case WI_HE_UNIFORM:
            val = sqrt(6.0 / (double)(obj->input_size));
            for (int i = 0; i < (obj->gamma).size; i++) {
                (obj->gamma).buffer[i] = random_uniform(-val, val * 2.0);
            }
            break;
        case WI_HE_NORMAL:
            val = sqrt(2.0 / (double)(obj->input_size));
            for (int i = 0; i < (obj->gamma).size; i++) {
                (obj->gamma).buffer[i] = random_normal(0.0, val);
            }
            break;
        default:
            LAST_ERROR = "Invalid gamma initializer.";
            return 0;
    }

    // Initialize beta.
    switch (beta_type) {
        case BI_ZEROS:
            for (int i = 0; i < (obj->beta).size; i++) {
                (obj->beta).buffer[i] = 0.0;
            }
            break;
        case BI_ONES:
            for (int i = 0; i < (obj->beta).size; i++) {
                (obj->beta).buffer[i] = 1.0;
            }
            break;
        default:
            LAST_ERROR = "Invalid beta initializer.";
            return 0;
    }

    return 1;
}

// Set the layer hyper parameters.
void layer_normalization_set_values(struct layer_normalization *obj, double epsilon, double momentum) {
    obj->epsilon = epsilon;
    obj->momentum = momentum;
}

// Perform a forward pass on the layer.
void layer_normalization_forward(struct layer_normalization *obj) {
    double mean, variance;
    for (int i = 0; i < obj->input_size; i++) {
        // Calculate the mean.
        mean = 0.0;
        for (int j = 0; j < obj->input->n_rows; j++) {
            mean += obj->input->buffer[j * obj->input_size + i];
        }
        mean /= (double)obj->input->n_rows;
        obj->mean.buffer[i] = mean;
        
        // Calculate the variance.
        variance = 0.0;
        for (int j = 0; j < obj->input->n_rows; j++) {
            variance += pow(obj->input->buffer[j * obj->input_size + i] - mean, 2.0);
        }
        variance /= (double)obj->input->n_rows;
        obj->variance.buffer[i] = variance;

        // Apply the normalization and affine transformation.
        for (int j = 0; j < obj->input->n_rows; j++) {
            obj->output->buffer[j * obj->input_size + i] = obj->gamma.buffer[i] * (obj->input->buffer[j * obj->input_size + i] - mean) / sqrt(variance + obj->epsilon) + obj->beta.buffer[i];
        }

        // Save the running mean and variance.
        obj->running_mean.buffer[i] = obj->momentum * obj->running_mean.buffer[i] + (1.0 - obj->momentum) * mean;
        obj->running_variance.buffer[i] = obj->momentum * obj->running_variance.buffer[i] + (1.0 - obj->momentum) * variance;
    }
}

// Perform a forward pass on the layer.
void layer_normalization_forward_predict(struct layer_normalization *obj) {
    for (int i = 0; i < obj->input_size; i++) {
        for (int j = 0; j < obj->input->n_rows; j++) {
            obj->output->buffer[j * obj->input_size + i] = obj->gamma.buffer[i] * (obj->input->buffer[j * obj->input_size + i] - obj->running_mean.buffer[i]) / sqrt(obj->running_variance.buffer[i] + obj->epsilon) + obj->beta.buffer[i];
        }
    }
}

// Perform a backward pass on the layer.
void layer_normalization_backward(struct layer_normalization *obj) {
    // Calculate d_inputs, d_gamma, and d_beta.
    double t, sum_d_outputs, sum_adjusted_d_outputs, sum_d_outputs_x_normalized, cached_adjusted_d_outputs;
    for (int i = 0; i < obj->input_size; i++) {
        t = 1.0 / sqrt(obj->variance.buffer[i] + obj->epsilon);
        sum_d_outputs = 0.0;
        sum_adjusted_d_outputs = 0.0;
        sum_d_outputs_x_normalized = 0.0;

        // sum_d_outputs = sum(d_outputs over axis 0).
        // sum_adjusted_d_outputs = sum(d_outputs * (x - mean) over axis 0).
        for (int j = 0; j < obj->input->n_rows; j++) {
            sum_d_outputs += obj->d_outputs->buffer[j * obj->input_size + i];
            cached_adjusted_d_outputs = obj->d_outputs->buffer[j * obj->input_size + i] * (obj->input->buffer[j * obj->input_size + i] - obj->mean.buffer[i]);
            sum_adjusted_d_outputs += cached_adjusted_d_outputs;
            sum_d_outputs_x_normalized += cached_adjusted_d_outputs / sqrt(obj->variance.buffer[i] + obj->epsilon);
        }

        // d_gamma = sum_d_outputs_x_normalized.
        obj->d_gamma.buffer[i] = sum_d_outputs_x_normalized;

        // d_beta = sum_d_outputs.
        obj->d_beta.buffer[i] = sum_d_outputs;

        for (int j = 0; j < obj->input->n_rows; j++) {
            double m = (double)obj->input->n_rows;
            
            // d_input = (gamma * t / m) * (m * d_output - sum_d_outputs - t ^ 2 * (x - mean) * sum_adjusted_d_outputs).
            obj->d_inputs->buffer[j * obj->input_size + i] = (obj->gamma.buffer[i] * t / m) * (m * obj->d_outputs->buffer[j * obj->input_size + i] - sum_d_outputs - t * t * (obj->input->buffer[j * obj->input_size + i] - obj->mean.buffer[i]) * sum_adjusted_d_outputs);
        }
    }
}