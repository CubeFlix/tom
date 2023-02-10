# Models

Models in `tom` use the `model` structure. A simple network can be made like so:

```
struct model m;

int main(void) {
    model_init(&m, 100);
    model_add_layer(&m, LAYER_DENSE, 10, 200);
    model_add_layer(&m, LAYER_RELU, 200, 200);
    model_set_loss(&m, LOSS_MSE);
    model_finalize(&m);

    model_free(&m);
}
```

## `model`

The model object.

```
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
```

### `int model_init(struct model *obj, int n_samples)`

Initialize an empty model object.

### `int model_free(struct model *obj)`

Free a model object. Free all the layers, optimizers, and matrices, along with the loss.

### `struct layer* model_add_layer(struct model *obj, enum layer_type type, int input_size, int output_size)`

Add and initialize a layer on the model. Returns the layer if successful.

### `struct layer* model_add_conv2d_layer(struct model* obj, int input_channels, int input_height, int input_width, int n_filters, int filter_size, int stride)`

Add a conv 2D layer without initializing it. Returns the layer if successful.

### `struct layer* model_add_maxpool2d_layer(struct model* obj, int input_channels, int input_height, int input_width, int pool_size, int stride)`

Add a max pooling 2D layer without initializing it. Returns the layer if successful. 

### `struct layer* model_add_padding2d_layer(struct model* obj, int input_channels, int input_height, int input_width, int padding_x, int padding_y)`

Add a padding 2D layer without initializing it. Returns the layer if successful.

### `void model_set_loss(struct model *obj, enum loss_type type)`

Set the model's loss.

### `int model_finalize(struct model *obj)`

Finalize and initialize the model.

### `int model_init_optimizers(struct model *obj, enum optimizer_type type, ...)`

Initialize optimizers on the model.

### `int model_predict(struct model* obj, struct matrix* X, struct matrix* Y)`

Predict. Takes an input and output matrix with any number of samples.

### `double model_calc_loss(struct model* obj, struct matrix* X, struct matrix* Y)`

Calculate model loss.

### `int model_train(struct model* obj, struct matrix* X, struct matrix* Y, int epochs, bool debug)`

Train the model.

### `int model_forward(struct model *obj, bool training)`

Perform a forward pass on the model.

### `int model_backward(struct model *obj)`

Perform a backward pass on the model.

### `int model_update(struct model* obj)`

Update each trainable layer in the model.
