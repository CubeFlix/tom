# Loss

## `loss_type`

Loss type enum.

```
enum loss_type {
    // Mean squared error.
    LOSS_MSE,

    // Mean absolute error.
    LOSS_MAE,

    // Cross-entropy loss.
    LOSS_CROSSENTROPY,

    // Binary cross-entropy loss.
    LOSS_BINARY_CROSSENTROPY
};
```

## `loss`

The generic loss object.

```
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
```

### `int loss_init(struct loss* obj, struct matrix* input, struct matrix* y, struct matrix* output, struct matrix* d_input)`

Initialize the loss object. The type should already be set. Requires the input, y, and output matrices, along with the input gradients. Returns `1` if successful, otherwise it returns `0`.

### `int loss_forward(struct loss* obj)`

Perform a forward pass on the loss. Returns `1` if successful, otherwise it returns `0`.

### `int loss_backward(struct loss* obj)`

Perform a backward pass on the loss. Returns `1` if successful, otherwise it returns `0`.

### `int loss_free(struct loss *obj)`

Free the loss object. Returns `1` if successful, otherwise it returns `0`.

## `loss_mse`

The Mean Squared Error (MSE) loss function. The output loss value is calculated as `sum((input - y) ** 2)`. The gradient on the inputs is calculated as `(input - y) * 2 / n_samples`. Note that a forward pass is not strictly required for training.

```
struct loss_mse {
    // The input and output size.
    int input_size, output_size;

    // The input, y, and output matrices.
    struct matrix *input, *y, *output;

    // Gradients on the inputs.
    struct matrix *d_inputs;
};
```

### `int loss_mse_init(struct loss_mse *obj, int input_size, struct matrix *input, struct matrix *y, struct matrix *output, struct matrix *d_inputs)`

Initialize an empty MSE loss object. Returns `1` if successful, otherwise it returns `0`.

### `double loss_mse_forward(struct loss_mse *obj)`

Perform a forward pass on the MSE loss. Returns the average loss over all samples.

### `void loss_mse_backward(struct loss_mse *obj)`

Perform a backward pass on the MSE loss.

## `loss_mae`

The Mean Absolute Error (MAE) loss function. The output loss value is calculated as `sum(abs(input - y))`. The gradient on the inputs is calculated as `sign(input - y)/ n_samples`. Note that a forward pass is not strictly required for training.

```
struct loss_mae {
    // The input and output size.
    int input_size, output_size;

    // The input, y, and output matrices.
    struct matrix *input, *y, *output;

    // Gradients on the inputs.
    struct matrix *d_inputs;
};
```

### `int loss_mae_init(struct loss_mae *obj, int input_size, struct matrix *input, struct matrix *y, struct matrix *output, struct matrix *d_inputs);`

Initialize an empty MAE loss object. Returns `1` if successful, otherwise it returns `0`.

### `double loss_mae_forward(struct loss_mae *obj)`

Perform a forward pass on the MAE loss. Returns the average loss over all samples.

### `void loss_mae_backward(struct loss_mae *obj)`

Perform a backward pass on the MAE loss.

## `loss_crossentropy`

The cross-entropy loss function. The forward pass is calculated as `-log(sum(y * input))`. The backward pass is calculated as `(-y / d_output) / n_samples`. The categorical cross-entropy `(softmax + cross-entropy)` backward pass can also be computed, which places the final gradients into `d_inputs`.

```
struct loss_crossentropy {
    // The input and output size.
    int input_size, output_size;

    // The input, y, and output matrices.
    struct matrix *input, *y, *output;

    // Gradients on the inputs.
    struct matrix *d_inputs;
};
```

### `int loss_crossentropy_init(struct loss_crossentropy *obj, int input_size, struct matrix *input, struct matrix *y, struct matrix *output, struct matrix *d_inputs)`

Initialize an empty cross-entropy loss object. Returns `1` if successful, otherwise it returns `0`.

### `double loss_crossentropy_forward(struct loss_crossentropy *obj)`

Perform a forward pass on the cross-entropy loss. Returns the average loss over all samples.

### `void loss_crossentropy_backward(struct loss_crossentropy *obj)`

Perform a backward pass on the cross-entropy loss.

### `void loss_crossentropy_backward_softmax(struct loss_crossentropy *obj)`

Perform a backward pass on the cross-entropy loss and the softmax activation.

## `loss_binary_crossentropy`

The binary cross-entropy loss function. The forward pass is calculated as `-(y * log(input) + (1-y) * log(1-input))`. The backward pass is calculated as `-(y / input - (1-y) / (1 - input))`. The forward pass returns the average loss over all samples.

```
struct loss_binary_crossentropy {
    // The input and output size.
    int input_size, output_size;

    // The input, y, and output matrices.
    struct matrix *input, *y, *output;

    // Gradients on the inputs.
    struct matrix *d_inputs;
};
```

### `int loss_binary_crossentropy_init(struct loss_binary_crossentropy *obj, int input_size, struct matrix *input, struct matrix *y, struct matrix *output, struct matrix *d_inputs)`

Initialize an empty binary cross-entropy loss object. Returns `1` if successful, otherwise it returns `0`.

### `double loss_binary_crossentropy_forward(struct loss_binary_crossentropy *obj)`

Perform a forward pass on the binary cross-entropy loss. Returns the average loss over all samples.

### `void loss_binary_crossentropy_backward(struct loss_binary_crossentropy *obj)`

Perform a backward pass on the binary cross-entropy loss.

