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

// Initialize a layer object. The layer should have its type, input size, and 
// output size set. Requires the input matrix and the gradients from the
// previous layer. Initalizes the layer object itself, along with the output
// matrix and output gradients.
int layer_init(struct layer *obj, int n_samples, struct matrix *inputs, struct matrix *d_prev) {
    // Create the new output matrix.
    struct matrix* current_output = calloc(1, sizeof(struct matrix));
    if (!matrix_init(current_output, n_samples, obj->output_size)) {
        return 0;
    }

    // Create the new gradient matrix.
    struct matrix* current_gradient = calloc(1, sizeof(struct matrix));
    if (!matrix_init(current_gradient, n_samples, obj->output_size)) {
        return 0;
    }

    switch (obj->type) {
    case LAYER_DENSE:;
        // Initialize the dense layer.
        struct layer_dense* dense = calloc(1, sizeof(struct layer_dense));
        if (!layer_dense_init(dense, obj->input_size, obj->output_size, inputs, current_output, current_gradient, d_prev)) {
            return 0;
        }
        obj->obj = dense;
        break;
    case LAYER_CONV2D:
        break;
    case LAYER_MAXPOOL2D:
        break;
    case LAYER_DROPOUT:
        break;
    case LAYER_SOFTMAX:;
        // Initialize the softmax activation layer.
        struct activation_softmax* softmax = calloc(1, sizeof(struct activation_softmax));
        if (!activation_softmax_init(softmax, obj->input_size, inputs, current_output, current_gradient, d_prev)) {
            return 0;
        }
        obj->obj = softmax;
        break;
    case LAYER_SIGMOID:
        break;
    case LAYER_RELU:
        break;
    default:
        LAST_ERROR = "Invalid layer type.";
        return 0;
    }

    // Set the values for the layer.
    obj->input = inputs;
    obj->output = current_output;
    obj->d_output = current_gradient;
    obj->d_input = d_prev;

    return 1;
}


// Initialize the layer values.

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
        free(obj->opt);
    }

    return 1;
}

// Initialize the loss object. The type should already be set. Requires the 
// input, y, and output matrices, along with the input gradients.
int loss_init(struct loss* obj, struct matrix* input, struct matrix* y,
              struct matrix* output, struct matrix* d_input) {
    // Set the loss values.
    obj->input = input;
    obj->y = y;
    obj->output = output;
    obj->d_input = d_input;
    
    switch (obj->loss.type) {
    case LOSS_MSE:;
        // Initialize the MSE loss.
        struct loss_mse* mse = calloc(1, sizeof(struct loss_mse));
        if (!loss_mse_init(mse, output->n_cols, input, y, output, d_input)) {
            return 0;
        }
        obj->obj = mse;
        break;
    case LOSS_CROSSENTROPY:
        // Initialize the crossentropy loss.
        struct loss_crossentropy* crossentropy = calloc(1, sizeof(struct loss_crossentropy));
        if (!loss_crossentropy_init(crossentropy, output->n_cols, input, y, output, d_input)) {
            return 0;
        }
        obj->obj = crossentropy;
        break;
    case LOSS_BINARY_CROSSENTROPY:
        break;
    default:
        LAST_ERROR = "Invalid loss type.";
        return 0;
    }

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

    // Free the object.
    free(obj->obj);

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

    // Free the loss output matrix.
    matrix_free(obj->loss_output);
    free(obj->loss_output);
    obj->loss_output = NULL;

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
        free(current);
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

// Set the layer's loss.
void model_set_loss(struct model *obj, enum loss_type type) {
    // Create the loss object.
    obj->loss.type = type;
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
        // Initialize the layer.
        if (!layer_init(current, obj->n_samples, obj->output, obj->last_gradient)) {
            return 0;
        }

        // Set the output and last gradient matricies.
        obj->output = current->output;
        obj->last_gradient = current->d_output;

        // Continue onto the next layer.
        current = current->next;
    } while (current != NULL);

    // Initialize the y matrix.
    obj->y = calloc(1, sizeof(struct matrix));
    if (!matrix_init(obj->y, obj->n_samples, obj->output->n_cols)) {
        return 0;
    }

    // Initialize the loss output.
    obj->loss_output = calloc(1, sizeof(struct matrix));
    if (!matrix_init(obj->loss_output, obj->n_samples, 1)) {
        return 0;
    }

    // Initialize the loss.
    if (obj->loss.type == LOSS_CROSSENTROPY && obj->last->type == LAYER_SOFTMAX) {
        // The loss should use the crossentropy softmax backward pass.
        if (!loss_init(&obj->loss, obj->output, obj->y, obj->loss_output, obj->last->d_input)) {
            return 0;
        }
    } else {
        // Initialize the loss normally.
        if (!loss_init(&obj->loss, obj->output, obj->y, obj->loss_output, obj->last_gradient)) {
            return 0;
        }
    }

    return 1;
}