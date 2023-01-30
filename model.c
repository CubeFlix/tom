// model.c
// Network model object.

#include <stdlib.h>
#include <stdio.h>

#include "model.h"
#include "matrix.h"
#include "dense.h"
#include "conv2d.h"
#include "maxpool2d.h"
#include "dropout.h"
#include "softmax.h"
#include "sgd.h"
#include "adam.h"
#include "mse.h"
#include "crossentropy.h"
#include "binary_crossentropy.h"

// Initialize an optimizer on the layer.
int layer_init_optimizer(struct layer *obj) {

}

// Free the layer, along with its matrices and optimizer. If a matrix is 
// already freed or not initialized, it will be skipped.
int layer_free(struct layer *obj) {
    switch (obj->type) {
        case LAYER_DENSE:
            layer_dense_free((struct layer_dense*)(obj->obj));
            break;
        case LAYER_CONV2D:
            layer_conv2d_free((struct layer_conv2d*)(obj->obj));
            break;
        case LAYER_MAXPOOL2D:
            layer_maxpool2d_free((struct layer_maxpool2d*)(obj->obj));
            break;
        case LAYER_DROPOUT:
            layer_dropout_free((struct layer_dropout*)(obj->obj));
            break;
        case LAYER_SOFTMAX:
            activation_softmax_free((struct activation_softmax*)(obj->obj));
            break;
        case LAYER_SIGMOID:
            break;
        case LAYER_RELU:
            break;
        default:
            LAST_ERROR = "Invalid layer type.";
            return 0;
    }

    // Free the layer itself.
    free(obj->obj);

    // If we have an optimizer, free it.
    if (obj->opt != NULL) {
        if (!optimizer_free(obj->opt)) {
            return 0;
        }
    }

    // Free ourselves.
    free(obj);

    return 1;
}

// Free the loss object.
int loss_free(struct loss *obj) {
    switch (obj->type) {
        case LOSS_MSE:
            break;
        case LOSS_CROSSENTROPY:
            break;
        case LOSS_BINARY_CROSSENTROPY:
            break;
        default:
            LAST_ERROR = "Invalid loss type.";
            return 0;
    }
    return 1;
}

// Free an optimizer object.
int optimizer_free(struct optimizer *obj) {
    switch (obj->type) {
        case OPTIMIZER_SGD:
            optimizer_sgd_free((struct optimizer_sgd*)(obj->obj));
        case OPTIMIZER_ADAM:
            optimizer_adam_free((struct optimizer_adam*)obj->obj);
        default:
            LAST_ERROR = "Invalid optimizer type.";
            return 0;
    }

    // Free ourselves.
    free(obj);

    return 1;
}

// Initialize an empty model object.
int model_init(struct model *obj, int n_samples) {
    // Set the number of samples.
    obj->n_samples = n_samples;

    return 1;
}

// Free a model object. Free all the layers, optimizers, and matrices, along
// with the loss.
int model_free(struct model *obj) {
    if (!obj->n_layers) {
        LAST_ERROR = "Model not initialized.";
        return 0;
    }
    
    // Free y matrix.
    matrix_free(obj->y);
    free(obj->y);
    obj->y = NULL;

    // Free the first input matrix.
    matrix_free(obj->input);
    free(obj->input);
    obj->input = NULL;

    // Free the first gradient.
    matrix_free(obj->first->d_input);
    free(obj->first->d_input);
    obj->first->d_input = NULL;

    // Free the matrices.
    struct layer *current = obj->first;
    do {
        // Free the output matrix.
        matrix_free(current->output);
        free(current->output);
        current->output = NULL;

        // Free the gradient.
        matrix_free(current->d_output);
        free(current->d_output);
        current->d_output = NULL;

        current = current->next;
    } while (current != NULL);

    // Free layers and optimizers.
    current = obj->first;
    struct layer *next;
    do {
        next = current->next;
        if (!layer_free(current)) {
            return 0;
        }
        current = next;
    } while (current != NULL);

    // Free the loss.
    return loss_free(&obj->loss);
}

// Add a layer without initializing it. Returns the layer if successful.
struct layer *model_add_layer(struct model *obj, enum layer_type type, int input_size, int output_size) {
    // Create the layer and set the values.
    struct layer *l = (struct layer*)calloc(1, sizeof(struct layer));
    l->prev = obj->last;
    obj->last = l;
    l->type = type;
    l->input_size = input_size;
    l->output_size = output_size;

    // If this is the first layer, set the first layer.
    if (!obj->n_layers) {
        obj->first = l;
    } else {
        // If this is not the first layer, set the "next" value for the 
        // previous layer.
        l->prev->next = l;
    }

    obj->n_layers++;

    return l;
}

// Finalize and initialize the model.
int model_finalize(struct model *obj) {
    // Ensure that there is at least one layer.
    if (!obj->n_layers) {
        LAST_ERROR = "No layers to initialize.";
        return 0;
    }

    // Initialize the input matrix.
    obj->input = calloc(1, sizeof(struct matrix));
    if (!matrix_init(obj->input, obj->n_samples, obj->first->input_size)) {
        return 0;
    }
    obj->output = obj->input;

    // Initialize the gradients on the first layer.
    obj->last_gradient = calloc(1, sizeof(struct matrix));
    if (!matrix_init(obj->last_gradient, obj->n_samples, obj->first->input_size)) {
        return 0;
    }
    
    // Initialize each layer.
    struct layer *current = obj->first;
    do {
        // Create the new output matrix.
        struct matrix *current_output = calloc(1, sizeof(struct matrix));

        // Create the new gradient matrix.
        struct matrix *current_gradient = calloc(1, sizeof(struct matrix));

        switch (current->type) {
            case LAYER_DENSE:
                // Initialize the dense layer.
                struct layer_dense *dense = calloc(1, sizeof(struct layer_dense));
                if (!matrix_init(current_output, obj->n_samples, current->output_size)) {
                    return 0;
                }
                if (!matrix_init(current_gradient, obj->n_samples, current->output_size)) {
                    return 0;
                }
                if (!layer_dense_init(dense, current->input_size, current->output_size, obj->output, current_output, current_gradient, obj->last_gradient)) {
                    return 0;
                }

                current->obj = dense;
                current->input = obj->output;
                current->output = current_output;
                current->d_output = current_gradient;
                current->d_input = obj->last_gradient;
                obj->output = current_output;
                obj->last_gradient = current_gradient;
                break;
            case LAYER_CONV2D:
                break;
            case LAYER_MAXPOOL2D:
                break;
            case LAYER_DROPOUT:
                break;
            case LAYER_SOFTMAX:
                break;
            case LAYER_SIGMOID:
                break;
            case LAYER_RELU:
                break;
            default:
                LAST_ERROR = "Invalid layer type.";
                return 0;
        }
        current = current->next;
    } while (current != NULL);

    // Initialize the y matrix.
    obj->y = calloc(1, sizeof(struct matrix));
    return matrix_init(obj->y, obj->n_samples, obj->output->n_cols);
}