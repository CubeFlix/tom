// model.h
// Network model object.

#ifndef MODEL_H
#define MODEL_H

#include <stdbool.h>
#include <stdarg.h>

#include "matrix.h"
#include "declspec.h"

extern char *LAST_ERROR;

#define IS_CROSSENTROPY_SOFTMAX(obj) (obj->loss.type == LOSS_CROSSENTROPY && obj->last->type == LAYER_SOFTMAX)

// Optimizer type enum.
enum optimizer_type {
    // Stochastic gradient descent.
    OPTIMIZER_SGD,

    // Adam optimizer.
    OPTIMIZER_ADAM,

    // RMSProp optimizer.
    OPTIMIZER_RMSPROP
};

// The generic optimizer object.
struct optimizer {
    // The optimizer type.
    enum optimizer_type type;

    // The current iteration.
    int iter;

    // The optimizer object.
    void* obj;
};

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
	
	// Leaky RELU layer.
	LAYER_LEAKY_RELU,

    // Sigmoid layer.
    LAYER_SIGMOID,

    // Softmax layer.
    LAYER_SOFTMAX,
	
	// Tanh layer.
	LAYER_TANH
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
    
    // Optional dimensions for 2D layers.
    int input_channels, input_height, input_width;
    int output_channels, output_height, output_width;

    // Optional parameters for conv and max pooling 2D layers.
    int filter_size, stride;
};

// Initialize a layer object. The layer should have its type, input size, and 
// output size set. Requires the input matrix and the gradients from the
// previous layer. Initializes the layer object itself, along with the output
// matrix and output gradients.
extern TOM_API int layer_init(struct layer* obj, int n_samples, struct matrix* inputs,
               struct matrix* d_prev);

// Initialize the layer's optimizer. Requires an optimizer type and variable
// args list, which will be used by the function.
extern TOM_API int layer_init_optimizer(struct layer* obj, enum optimizer_type type,
                         va_list ap);

// Perform a forward pass on the layer.
extern TOM_API int layer_forward(struct layer* obj, bool training);

// Perform a backward pass on the layer.
extern TOM_API int layer_backward(struct layer* obj);

// Free the layer, along with its matrices and optimizer. If a matrix is 
// already freed or not initialized, it will be skipped.
extern TOM_API int layer_free(struct layer *obj);

// Perform an update on the layer's optimizer.
extern TOM_API int layer_update(struct layer* obj);

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

    // Current average batch loss value.
    double batch_loss;
};

// Initialize the loss object. The type should already be set. Requires the 
// input, y, and output matrices, along with the input gradients.
extern TOM_API int loss_init(struct loss* obj, struct matrix* input, struct matrix* y,
              struct matrix* output, struct matrix* d_input);

// Perform a forward pass on the loss.
extern TOM_API int loss_forward(struct loss* obj);

// Perform a backward pass on the loss.
extern TOM_API int loss_backward(struct loss* obj);

// Free the loss object.
extern TOM_API int loss_free(struct loss *obj);

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
extern TOM_API int model_init(struct model *obj, int n_samples);

// Free a model object. Free all the layers, optimizers, and matrices, along
// with the loss.
extern TOM_API int model_free(struct model *obj);

// Add and initialize a layer.
extern TOM_API struct layer* model_add_layer(struct model *obj, enum layer_type type, 
                              int input_size, int output_size);

// Add a conv 2D layer without initializing it. Returns the layer if 
// successful.
extern TOM_API struct layer* model_add_conv2d_layer(struct model* obj,
    int input_channels, int input_height,
    int input_width, int n_filters,
    int filter_size, int stride);

// Add a max pooling 2D layer without initializing it. Returns the layer if 
// successful.
extern TOM_API struct layer* model_add_maxpool2d_layer(struct model* obj,
    int input_channels, int input_height,
    int input_width, int pool_size,
    int stride);

// Set the layer's loss.
extern TOM_API void model_set_loss(struct model *obj, enum loss_type type) ;

// Finalize and initialize the model.
extern TOM_API int model_finalize(struct model *obj);

// Initialize optimizers on the model.
extern TOM_API int model_init_optimizers(struct model *obj, enum optimizer_type type, ...);

// Predict. Takes an input and output matrix with any number of samples.
extern TOM_API int model_predict(struct model* obj, struct matrix* X, struct matrix* Y);

// Calculate model loss.
extern TOM_API double model_calc_loss(struct model* obj, struct matrix* X, struct matrix* Y);

// Train the model.
extern TOM_API int model_train(struct model* obj, struct matrix* X, struct matrix* Y, 
                int epochs, bool debug);

// Perform a forward pass on the model.
extern TOM_API int model_forward(struct model *obj, bool training);

// Perform a backward pass on the model.
extern TOM_API int model_backward(struct model *obj);

// Update each trainable layer in the model.
extern TOM_API int model_update(struct model* obj);

#endif
