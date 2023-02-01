// model.h
// Network model object.

#ifndef MODEL_H
#define MODEL_H

#include <stdbool.h>

#include "matrix.h"

extern char *LAST_ERROR;

#define IS_CROSSENTROPY_SOFTMAX(obj) (obj->loss.type == LOSS_CROSSENTROPY && obj->last->type == LAYER_SOFTMAX)

// Optimizer type enum.
enum optimizer_type {
    // Stochastic gradient descent.
    OPTIMIZER_SGD,

    // Adam optimizer.
    OPTIMIZER_ADAM
};

// The generic optimizer object.
struct optimizer {
    // The optimizer type.
    enum optimizer_type type;

    // The optimizer object.
    void* obj;
};

// Free an optimizer object.
int optimizer_free(struct optimizer* obj);

// Layer type enum.
enum layer_type {
    // Dense layer.
    LAYER_DENSE,

    // 2D conv layer.
    LAYER_CONV2D,

    // 2D max pool layer.
    LAYER_MAXPOOL2D,

    // Dropout layer.
    LAYER_DROPOUT,

    // RELU layer.
    LAYER_RELU,

    // Sigmoid layer.
    LAYER_SIGMOID,

    // Softmax layer.
    LAYER_SOFTMAX
};

// The generic layer object. 
struct layer {
    // Next and previous layers.
    struct layer *prev, *next;

    // The layer type.
    enum layer_type type;

    // The layer object.
    void *obj;

    // If the layer is trainable.
    bool trainable;

    // The (optional) optimizer.
    struct optimizer opt;

    // Matrices for the layer. We store the input and output matrices, along
    // with the input and output gradients.
    struct matrix *input, *output;
    struct matrix *d_output, *d_input;

    // The input and output size.
    int input_size, output_size;
};

// Free the layer, along with its matrices and optimizer. If a matrix is 
// already freed or not initialized, it will be skipped.
int layer_free(struct layer *obj);

// Loss type enum.
enum loss_type {
    // Mean squared error.
    LOSS_MSE,

    // Cross-entropy loss.
    LOSS_CROSSENTROPY,

    // Binary cross-entropy loss.
    LOSS_BINARY_CROSSENTROPY
};

// The generic loss object.
struct loss {
    // The loss type.
    enum loss_type type;

    // The loss object.
    void *obj;

    // Matrices for the loss. We store the input and output matrices, along 
    // with the gradient and y matrix.
    struct matrix *input, *output;
    struct matrix *d_input;
    struct matrix *y;
};

// Free the loss object.
int loss_free(struct loss *obj);

// The model object.
struct model {
    // First and last layers.
    struct layer *first, *last;

    // Number of layers.
    int n_layers;

    // Number of samples.
    int n_samples;

    // Loss object.
    struct loss loss;

    // We store the input, output and y matrices.
    struct matrix *input, *output, *y;

    // Loss output.
    struct matrix *loss_output;

    // Store the last gradient.
    struct matrix *last_gradient;
};

// Initialize an empty model object.
int model_init(struct model *obj, int n_samples);

// Free a model object. Free all the layers, optimizers, and matrices, along
// with the loss.
int model_free(struct model *obj);

// Add and initialize a layer.
struct layer *model_add_layer(struct model *obj, enum layer_type type, int input_size, int output_size);

// Add and initialize a 2D conv or max pooling layer.
int model_add_2d_layer(struct model *obj);

// Set the layer's loss.
void model_set_loss(struct model *obj, enum loss_type type) ;

// Finalize and initialize the model.
int model_finalize(struct model *obj);

// Initialize optimizers on the model.
int model_init_optimizers(struct model *obj, enum optimizer_type type, ...);

// Perform a backward pass on the model.
int model_backward(struct model *obj);

// Perform a backward pass on the model.
int model_backward(struct model *obj);

#endif