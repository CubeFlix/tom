// model.c
// Network model object.

#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>

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
        free(current_output);
        return 0;
    }

    // Create the new gradient matrix.
    struct matrix* current_gradient = calloc(1, sizeof(struct matrix));
    if (!matrix_init(current_gradient, n_samples, obj->output_size)) {
        free(current_gradient);
        return 0;
    }

    switch (obj->type) {
    case LAYER_DENSE:;
        // Initialize the dense layer.
        struct layer_dense* dense = calloc(1, sizeof(struct layer_dense));
        if (!layer_dense_init(dense, obj->input_size, obj->output_size, inputs, current_output, current_gradient, d_prev)) {
            free(dense);
            return 0;
        }
        obj->trainable = true;
        obj->obj = dense;
        break;
    case LAYER_CONV2D:
        obj->trainable = true;
        break;
    case LAYER_MAXPOOL2D:
        break;
    case LAYER_DROPOUT:
        break;
    case LAYER_SOFTMAX:;
        // Initialize the softmax activation layer.
        struct activation_softmax* softmax = calloc(1, sizeof(struct activation_softmax));
        if (!activation_softmax_init(softmax, obj->input_size, inputs, current_output, current_gradient, d_prev)) {
            free(softmax);
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

// Initialize the layer's optimizer. Requires an optimizer type and variable
// args list, which will be used by the function.
int layer_init_optimizer(struct layer* obj, enum optimizer_type type, va_list ap) {
    obj->opt.type = type;

    switch (obj->type) {
    case LAYER_DENSE:
        switch (type) {
        case OPTIMIZER_SGD:;
            // Stochastic gradient descent.
            struct optimizer_sgd* sgd = calloc(1, sizeof(struct optimizer_sgd));
            obj->opt.obj = sgd;
            if (!optimizer_sgd_init(sgd, obj->obj, va_arg(ap, double), va_arg(ap, double), va_arg(ap, double))) {
                free(sgd);
                return 0;
            }
            break;
        case OPTIMIZER_ADAM:;
            // Adam.
            struct optimizer_adam* adam = calloc(1, sizeof(struct optimizer_adam));
            obj->opt.obj = adam;
            if (!optimizer_adam_init(adam, obj->obj, va_arg(ap, double), va_arg(ap, double), va_arg(ap, double), va_arg(ap, double), va_arg(ap, double))) {
                free(adam);
                return 0;
            }
            break;
        default:
            LAST_ERROR = "Invalid optimizer type.";
            return 0;
        }
        break;
    case LAYER_CONV2D:
        break;
    default:
        LAST_ERROR = "Layer is untrainable; cannot initialize optimizer.";
        return 0;
    }

    return 1;
}

// Perform a forward pass on the layer.
int layer_forward(struct layer *obj, bool training) {
    switch (obj->type) {
    case LAYER_DENSE:;
        layer_dense_forward(obj->obj);
        break;
    case LAYER_CONV2D:
        break;
    case LAYER_MAXPOOL2D:
        break;
    case LAYER_DROPOUT:
        if (training) {
            layer_dropout_forward(obj->obj);
        } else {
            layer_dropout_forward_predict(obj->obj);
        }
        break;
    case LAYER_SOFTMAX:
        // Initialize the softmax activation layer.
        activation_softmax_forward_stable(obj->obj);
        break;
    case LAYER_SIGMOID:
        break;
    case LAYER_RELU:
        break;
    default:
        LAST_ERROR = "Invalid layer type.";
        return 0;
    }
    return 1;
}

// Perform a backward pass on the layer.
int layer_backward(struct layer *obj) {
    switch (obj->type) {
    case LAYER_DENSE:;
        layer_dense_backward(obj->obj);
        break;
    case LAYER_CONV2D:
        break;
    case LAYER_MAXPOOL2D:
        break;
    case LAYER_DROPOUT:
        layer_dropout_backward(obj->obj);
        break;
    case LAYER_SOFTMAX:
        // Initialize the softmax activation layer.
        activation_softmax_backward(obj->obj);
        break;
    case LAYER_SIGMOID:
        break;
    case LAYER_RELU:
        break;
    default:
        LAST_ERROR = "Invalid layer type.";
        return 0;
    }
    return 1;
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

    // Free the optimizer.
    if (!optimizer_free(&obj->opt)) {
        return 0;
    }

    return 1;
}

// Initialize the loss object. The type should already be set. Requires the 
// input, y, and output matrices, along with the input gradients.
int loss_init(struct loss* obj, struct matrix* input, struct matrix* y,
              struct matrix* output, struct matrix* d_input) {
    // Set the loss values.
    obj->input = input;
    obj->output = output;
    obj->d_input = d_input;
    obj->y = y;

    switch (obj->type) {
    case LOSS_MSE:;
        // Initialize the MSE loss.
        struct loss_mse *mse = calloc(1, sizeof(struct loss_mse));
        if (!loss_mse_init(mse, input->n_cols, input, y, output, d_input)) {
            free(mse);
            return 0;
        }
        obj->obj = mse;
        break;
    case LOSS_CROSSENTROPY:;
        // Initialize the crossentropy loss.
        struct loss_crossentropy *crossentropy = calloc(1, sizeof(struct loss_crossentropy));
        if (!loss_crossentropy_init(crossentropy, input->n_cols, input, y, output, d_input)) {
            free(crossentropy);
            return 0;
        }
        obj->obj = crossentropy;
        break;
    case LOSS_BINARY_CROSSENTROPY:;
        // Initialize the binary crossentropy loss.
        struct loss_binary_crossentropy *binary_crossentropy = calloc(1, sizeof(struct loss_binary_crossentropy));
        if (!loss_binary_crossentropy_init(binary_crossentropy, input->n_cols, input, y, output, d_input)) {
            free(binary_crossentropy);
            return 0;
        }
        obj->obj = binary_crossentropy;
        break;
    default:
        LAST_ERROR = "Invalid loss type.";
        return 0;
    }

    return 1;
}

// Perform a forward pass on the loss.
int loss_forward(struct loss *obj) {
    switch (obj->type) {
    case LOSS_MSE:
        loss_mse_forward(obj->obj);
        break;
    case LOSS_CROSSENTROPY:
        loss_crossentropy_forward(obj->obj);
        break;
    case LOSS_BINARY_CROSSENTROPY:
        loss_binary_crossentropy_forward(obj->obj);
        break;
    default:
        LAST_ERROR = "Invalid loss type.";
        return 0;
    }

    return 1;
}

// Perform a backward pass on the loss.
int loss_backward(struct loss *obj) {
    switch (obj->type) {
    case LOSS_MSE:
        loss_mse_backward(obj->obj);
        break;
    case LOSS_CROSSENTROPY:
        loss_crossentropy_backward(obj->obj);
        break;
    case LOSS_BINARY_CROSSENTROPY:
        loss_binary_crossentropy_backward(obj->obj);
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
    if (obj->obj != NULL) {
        switch (obj->type) {
        case OPTIMIZER_SGD:
            optimizer_sgd_free((struct optimizer_sgd*)(obj->obj));
            break;
        case OPTIMIZER_ADAM:
            optimizer_adam_free((struct optimizer_adam*)(obj->obj));
            break;
        default:
            LAST_ERROR = "Invalid optimizer type.";
            return 0;
        }
    }

    // Free the optimizer itself.
    free(obj->obj);

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
        free(obj->input);
        return 0;
    }
    obj->output = obj->input;

    // Initialize the gradients on the first layer.
    obj->last_gradient = calloc(1, sizeof(struct matrix));
    if (!matrix_init(obj->last_gradient, obj->n_samples, obj->first->input_size)) {
        free(obj->last_gradient);
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
        free(obj->y);
        return 0;
    }

    // Initialize the loss output.
    obj->loss_output = calloc(1, sizeof(struct matrix));
    if (!matrix_init(obj->loss_output, obj->n_samples, 1)) {
        free(obj->loss_output);
        return 0;
    }

    // Initialize the loss.
    if (IS_CROSSENTROPY_SOFTMAX(obj)) {
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

// Initialize optimizers on the model.
int model_init_optimizers(struct model* obj, enum optimizer_type type, ...) {
    int n_args;
    va_list ap;

    // Get the number of arguments.
    switch (type) {
    case OPTIMIZER_SGD:
        n_args = 3;
        break;
    case OPTIMIZER_ADAM:
        n_args = 5;
        break;
    default:
        LAST_ERROR = "Invalid optimizer type.";
        return 0;
    }

    va_start(ap, n_args);

    // Loop over each layer.
    struct layer* current = obj->first;
    do {
        if (current->trainable) {
            // Make a copy of the arguments.
            va_list ap_copy;
            va_copy(ap_copy, ap);

            // Initialize the optimizer.
            if (!layer_init_optimizer(current, type, ap_copy)) {
                return 0;
            }
        }
        current = current->next;
    } while (current != NULL);

    va_end(ap);

    return 1;
}

// Perform a forward pass on the entire model.
int model_forward(struct model *obj, bool training) {
    // Perform the forward pass through each layer.
    struct layer *current = obj->first;
    do {
        if (!layer_forward(current, training)) {
            return 0;
        }
        current = current->next;
    } while (current != NULL);

    // Perform the forward pass through the loss.
    return loss_forward(&obj->loss);
}

// Perform a backward pass on the model.
int model_backward(struct model *obj) {
    // Perform the backward pass through the loss.
    if (!loss_backward(&obj->loss)) {
        return 0;
    }

    // Perform the backward pass through each layer.
    struct layer *current = obj->last;
    if (IS_CROSSENTROPY_SOFTMAX(obj)) {
        // Skip the softmax layer.
        current = current->prev;
    }

    do {
        if (!layer_backward(current)) {
            return 0;
        }
        current = current->prev;
    } while (current != NULL);
    
    return 1;
}