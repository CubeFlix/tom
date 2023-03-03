# Layers

Layers in `tom` are represented with the `layer` structure. Each layer stores a pointer to its input matrix, output matrix, input gradient matrix, and output gradient matrix. The layer also stores its type, along with a pointer to its underlying layer object.

## `layer_type`
```
// Layer type enum.
enum layer_type {
    // Dense layer.
    LAYER_DENSE,

    // 2D conv layer.
    LAYER_CONV2D,

    // 2D max pool layer.
    LAYER_MAXPOOL2D,

    // 2D padding layer.
    LAYER_PADDING2D,

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
```

## `layer`

```
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

    // Optional parameters for padding 2D layers.
    int padding_x, padding_y;
};
```

### `int layer_init(struct layer *obj, int n_samples, struct matrix *inputs, struct matrix *d_prev)`

Initialize a layer object. The layer should have its type, input size, and output size set. Requires the input matrix and the gradients from the previous layer. Initializes the underlying layer object itself, along with the output matrix and output gradients. Returns `1` if successful, otherwise it returns `0`.

### `int layer_init_optimizer(struct layer *obj, enum optimizer_type type, va_list ap)`

Initialize the layer's optimizer. Requires an optimizer type and variable args list, which will be used by the function. The `ap` parameter should store the arguments to be passed to the optimizer's initialization function. Returns `1` if successful, otherwise it returns `0`.

### `int layer_forward(struct layer *obj, bool training)`

Perform a forward pass on the layer. Requires the `input` matrix to be set. `training` applies only to dropout layers; if `training` is `false`, the dropout layer will do nothing to the inputs. Returns `1` if successful, otherwise it returns `0`.

### `int layer_backward(struct layer *obj)`

Perform a backward pass on the layer. Requires the `d_output` matrix to be set. Returns `1` if successful, otherwise it returns `0`.

### `int layer_free(struct layer *obj)`

Free the layer, along with its matrices and optimizer. If a matrix is already freed or not initialized, it will be skipped. Returns `1` if successful, otherwise it returns `0`.

### `int layer_update(struct layer *obj)`

Perform an update on the layer's optimizer. Returns `1` if successful, otherwise it returns `0`.

## `weight_initializer`
Dense and conv 2D layer weight initializers. 

```
enum weight_initializer {
    // Generates zeros (0.0).
    WI_ZEROS,

    // Generates ones (1.0).
    WI_ONES,

    // Generates numbers from a uniform distribution [-1.0, 1.0]
    WI_RANDOM_UNIFORM,

    // Generates numbers from a normal distribution, centered at 0.0 with a 
    // standard deviation of 1.0.
    WI_RANDOM_NORMAL,

    // Generates numbers using Glorot normal initialization.
    WI_GLOROT_NORMAL,

    // Generates numbers using Glorot uniform initialization.
    WI_GLOROT_UNIFORM,

    // Generates numbers using He normal initialization.
    WI_HE_NORMAL,

    // Generates numbers using He uniform initialization.
    WI_HE_UNIFORM
};
```

## `bias_initializer`

Dense and conv 2D layer bias initializers.

```
enum bias_initializer {
    BI_ZEROS,
    BI_ONES
};
```

## `layer_dense`

The standard fully-connected dense layer. The layer stores the input and output size, along with its weights, biases, and gradients. On a forward pass, the layer performs the operation `X*W + b` on the input matrix and places its output in the output matrix. On a backward pass, the gradient is calculated based on the gradients of the outputs, calculated by the following layer. The calculated gradients are then stored for the optimizer, and the gradients on the inputs are then passed to the preceding layer. Because the inputs and outputs are shared between layers, they are not initialized by the layer. Similarly, with the gradients of the outputs and inputs, they are passed in on initialization as pointers, and are not stored within the layer itself. However, the weights and biases, along with the gradients on the weights and biases, are initialized and fully managed by the layer. The dense layer supports L1 and L2 weight and bias regularization.

```
struct layer_dense {
    // The input and output size.
    int input_size, output_size;

    // The input and output matrices.
    struct matrix *input, *output;
    
    // The weights and biases.
    struct matrix weights, biases;

    // Gradients on the outputs, inputs, weights, and biases, respectively.
    struct matrix *d_outputs, *d_inputs, d_weights, d_biases;

    // Regularization values.
    double l1_weights, l1_biases, l2_weights, l2_biases;
};
```

### `int layer_dense_init(struct layer_dense *obj, int input_size, int output_size, struct matrix *input, struct matrix *output, struct matrix *d_outputs, struct matrix *d_inputs)`

Initialize an empty dense layer object. Returns `1` if successful, otherwise it returns `0`.

### `int layer_dense_init_values(struct layer_dense *obj, enum weight_initializer wi_type, enum bias_initializer bi_type)`

Initialize the weights and biases for a dense layer object. Requires the layer to be initialized first, along with a valid weight initializer type and bias initializer type. Returns `1` if successful, otherwise it returns `0`.

### `void layer_dense_init_regularization(struct layer_dense *obj, double l1_weights, double l2_weights, double l1_biases, double l2_biases)`

Initialize the regularization values. Dense layers support L1 and L2 regularization.

### `void layer_dense_free(struct layer_dense *obj)`

Free the matrices owned by the dense layer.

### `void layer_dense_forward(struct layer_dense *obj)`

Perform a forward pass on the dense layer.

### `void layer_dense_backward(struct layer_dense *obj)`

Perform a backward pass on the dense layer.

### `double layer_dense_calculate_regularization(struct layer_dense *obj)`

Calculate the total regularization loss for the dense layer.

## `layer_conv2d`

The 2D conv layer. The dimensions of the input matrix are `(n_samples, n_channels * input_height * input_width)`. The dimensions of the output matrix are `(n_samples, n_filters * output_height * output_width)`. The output height is `(input_height - filter_size)/stride + 1`. The output width is `(input_width - filter_size)/stride + 1`. The total number of filters is `n_filters * n_channels`.

```
struct layer_conv2d {
    // The input and output dimensions.
    int n_channels, input_height, input_width, output_height, output_width;

    // The hyperparameter values for the conv layer.
    int n_filters, filter_size, stride;

    // The input and output matrices.
    struct matrix *input, *output;
    
    // The weights and biases.
    struct matrix weights, biases;

    // Gradients on the outputs, inputs, weights, and biases, respectively.
    struct matrix *d_outputs, *d_inputs, d_weights, d_biases;
};
```

### `CALC_CONV2D_OUTPUT_DIM(dim, filter_size, stride)`
Calculate an output dimension for a conv 2D layer. Returns `((dim - filter_size) / stride + 1)`.

### `int layer_conv2d_init(struct layer_conv2d *obj, int n_channels, int input_height, int input_width, int n_filters, int filter_size, int stride, struct matrix *input, struct matrix *output, struct matrix *d_outputs, struct matrix *d_inputs)`

Initialize an empty conv 2D layer object. Returns `1` if successful, otherwise it returns `0`.

### `int layer_conv2d_init_values(struct layer_conv2d *obj, enum weight_initializer wi_type, enum bias_initializer bi_type)`

Initialize the weights and biases for a conv 2D layer object. Requires the layer to be initialized first, along with a valid weight initializer type and bias initializer type. Returns `1` if successful, otherwise it returns `0`.

### `void layer_conv2d_free(struct layer_conv2d *obj)`

Free the matrices owned by the conv 2D layer.

### `void layer_conv2d_forward(struct layer_conv2d *obj)`

Perform a forward pass on the conv 2D layer.

### `void layer_conv2d_backward(struct layer_conv2d *obj)`

Perform a backward pass on the conv 2D layer.

## `layer_maxpool2d`

The 2D max pooling layer.

```
struct layer_maxpool2d {
    // The input and output dimensions.
    int n_channels, input_height, input_width, output_height, output_width;

    // The hyperparameter values for the pooling layer.
    int pool_size, stride;

    // The input and output matrices.
    struct matrix *input, *output;

    // Max pooling cache. For each input value, the value is 1.0 where the 
    // value is the maximum, while the value is 0.0 where the value is not.
    struct matrix cache;
    
    // Gradients on the outputs and inputs, respectively.
    struct matrix *d_outputs, *d_inputs;
};
```

### `CALC_MAXPOOL2D_OUTPUT_DIM(dim, pool_size, stride)`
Calculate an output dimension for a max pooling 2D layer. Returns `((dim - pool_size) / stride + 1)`.

### `int layer_maxpool2d_init(struct layer_maxpool2d *obj, int n_channels, int input_height, int input_width, int pool_size, int stride, struct matrix *input, struct matrix *output, struct matrix *d_outputs, struct matrix *d_inputs)`
Initialize an empty max pooling 2D layer object. Returns `1` if successful, otherwise it returns `0`.

### `void layer_maxpool2d_free(struct layer_maxpool2d *obj)`

Free the matrices owned by the max pooling 2D layer.

### `void layer_maxpool2d_forward(struct layer_maxpool2d *obj)`

Perform a forward pass on the max pooling 2D layer.

### `void layer_maxpool2d_backward(struct layer_maxpool2d *obj)`

Perform a backward pass on the max pooling 2D layer.

## `padding_type`

Padding type enum.

```
enum padding_type {
    // Zero padding, or "same padding".
    PADDING_ZERO,

    // Symmetric padding.
    PADDING_SYMMETRIC,

    // Reflection padding.
    PADDING_REFLECTION
};
```

## `layer_padding2d`

The 2D padding layer.

```
struct layer_padding2d {
    // The input and output dimensions.
    int n_channels, input_height, input_width, output_height, output_width;
    
    // The hyperparameter values for the padding layer. The padding is applied
    // twice to each dimension, on both sides.
    int padding_x, padding_y;
    enum padding_type type;

    // The input and output matrices.
    struct matrix *input, *output;

    // Padding output cache. For each output value, it stores the index of the 
    // corresponding input value. The cache only needs to be calculated once,
    // on the first forward pass.
    int *output_cache;

    // Padding gradient cache. For each input value, the value is 1.0 
    // plus 1.0 multiplied by the number of times said input value appears in
    // the output padding. For example, if a particular input value is repeated
    // twice in the output padding, its corresponding gradient cache will be 
    // 3.0. For zero-padding, all values will be 1.0. The cache only needs
    // to be calculated once, on the first forward pass.
    struct matrix grad_cache;

    bool has_caches;
    
    // Gradients on the outputs and inputs, respectively.
    struct matrix *d_outputs, *d_inputs;
};
```

### `CALC_PADDING2D_OUTPUT_DIM(dim, padding)`
Calculate an output dimension for a max pooling 2D layer. Returns `(dim + padding * 2)`.

### `int layer_padding2d_init(struct layer_padding2d *obj, int n_channels, int input_height, int input_width, int padding_x, int padding_y, enum padding_type type, struct matrix *input, struct matrix *output, struct matrix *d_outputs, struct matrix *d_inputs)`

Initialize an empty padding 2D layer object. Returns `1` if successful, otherwise it returns `0`.

### `void layer_padding2d_free(struct layer_padding2d *obj)`

Free the matrices owned by the padding 2D layer.

### `void layer_padding2d_set_type(struct layer_padding2d *obj, enum padding_type type)`

Set the padding type.

### `int layer_padding2d_recalculate_caches(struct layer_padding2d *obj)`

Recalculate the caches for the padding 2D layer. Returns `1` if successful, otherwise it returns `0`.

### `int layer_padding2d_forward(struct layer_padding2d *obj)`

Perform a forward pass on the padding 2D layer. Returns `1` if successful, otherwise it returns `0`.

### `void layer_padding2d_backward(struct layer_padding2d *obj)`

Perform a backward pass on the padding 2D layer.

## `layer_dropout`

The dropout layer. 

```
struct layer_dropout {
    // The input and output size.
    int input_size, output_size;

    // The input and output matrices.
    struct matrix *input, *output;
    
    // The current binary mask.
    struct matrix mask;

    // Gradients on the outputs and inputs.
    struct matrix *d_outputs, *d_inputs;

    // Dropout rate.
    double rate;
};
```

### `int layer_dropout_init(struct layer_dropout *obj, int input_size, double rate, struct matrix *input, struct matrix *output, struct matrix *d_outputs, struct matrix *d_inputs)`

Initialize an empty dropout layer object. Returns `1` if successful, otherwise it returns `0`.

### `void layer_dropout_set_rate(struct layer_dropout *obj, double rate)`

Set the dropout layer's rate.

### `void layer_dropout_free(struct layer_dropout *obj)`

Free the matrices owned by the dropout layer.

### `void layer_dropout_forward(struct layer_dropout *obj)`

Perform a forward pass on the dropout layer.

### `void layer_dropout_forward_predict(struct layer_dropout *obj)`

Perform a forward pass on the dropout layer, without applying dropout.

### `void layer_dropout_backward(struct layer_dropout *obj)`

Perform a backward pass on the dropout layer.

## `activation_relu`

The Rectified Linear Unit (RELU) activation layer. The output is calculated as `1 * x` if `x > 0`, or `0` if `x <= 0`. The gradient is calculated as `d_output` if `x > 0`, or `0` if `x <= 0`.

```
struct activation_relu {
    // The input and output size.
    int input_size, output_size;

    // The input and output matrices.
    struct matrix *input, *output;

    // Gradients on the outputs and inputs.
    struct matrix *d_outputs, *d_inputs;
};
```

### `int activation_relu_init(struct activation_relu *obj, int input_size, struct matrix *input, struct matrix *output, struct matrix *d_outputs, struct matrix *d_inputs)`

Initialize an empty ReLU layer object. Returns `1` if successful, otherwise it returns `0`.

### `void activation_relu_forward(struct activation_relu *obj)`

Perform a forward pass on the ReLU layer.

### `void activation_relu_backward(struct activation_relu *obj)`

Perform a backward pass on the ReLU layer.

## `activation_leaky_relu`

The Leaky Rectified Linear Unit (RELU) activation layer. The output is calculated as `1 * x` if `x > 0`, or `rate * x` if `x <= 0`. The gradient is calculated as `d_output` if `x > 0`, or `rate * d_output` if `x <= 0`.

```
struct activation_leaky_relu {
    // The input and output size.
    int input_size, output_size;
	
	// The leaky RELU rate.
	double rate;

    // The input and output matrices.
    struct matrix *input, *output;

    // Gradients on the outputs and inputs.
    struct matrix *d_outputs, *d_inputs;
};
```

### `int activation_leaky_relu_init(struct activation_leaky_relu *obj, int input_size, double rate, struct matrix *input, struct matrix *output, struct matrix *d_outputs, struct matrix *d_inputs)`

Initialize an empty leaky ReLU layer object. Returns `1` if successful, otherwise it returns `0`.

### `void activation_leaky_relu_forward(struct activation_leaky_relu *obj)`

Perform a forward pass on the leaky ReLU layer.

### `void activation_leaky_relu_backward(struct activation_leaky_relu *obj)`

Perform a backward pass on the leaky ReLU layer.

## `layer_softmax`

The softmax activation function. The forward pass is calculated as `e^x/sum(e^x)`. The forward pass can be calculated in a numerically unstable method or numerically stable method. The backward pass is calculated as `(diag(yhat) - yhat^T * yhat)`.

```
struct activation_softmax {
    // The input and output size.
    int input_size, output_size;

    // The input and output matrices.
    struct matrix *input, *output;

    // Gradients on the outputs and inputs.
    struct matrix *d_outputs, *d_inputs;

    // Activation's internal Jacobian matrix.
    struct matrix jacobian;
};
```

### `activation_softmax_init(struct activation_softmax *obj, int input_size, struct matrix *input, struct matrix *output, struct matrix *d_outputs, struct matrix *d_inputs)`

Initialize an empty softmax layer object. Returns `1` if successful, otherwise it returns `0`.

### `void activation_softmax_free(struct activation_softmax *obj)`

Free the matrices owned by the softmax layer.

### `void activation_softmax_forward(struct activation_softmax *obj)`

Perform a forward pass on the softmax layer.

### `void activation_softmax_forward_stable(struct activation_softmax *obj)`

Perform a stable forward pass on the softmax layer.

### `void activation_softmax_backward(struct activation_softmax *obj)`

Perform a stable backward pass on the softmax layer.

## `layer_sigmoid`

The sigmoid activation function. The forward pass is calculated as `1 / (1 + e^(-x))`. The backward pass is calculated as `d_output * y * (1-y)`.

```
struct activation_sigmoid {
    // The input and output size.
    int input_size, output_size;

    // The input and output matrices.
    struct matrix *input, *output;

    // Gradients on the outputs and inputs.
    struct matrix *d_outputs, *d_inputs;
};
```

### `int activation_sigmoid_init(struct activation_sigmoid *obj, int input_size, struct matrix *input, struct matrix *output, struct matrix *d_outputs, struct matrix *d_inputs)`

Initialize an empty sigmoid layer object. Returns `1` if successful, otherwise it returns `0`.

### `void activation_sigmoid_forward(struct activation_sigmoid *obj)`

Perform a forward pass on the sigmoid layer.

### `void activation_sigmoid_backward(struct activation_sigmoid *obj)`

Perform a backward pass on the sigmoid layer.

## `layer_tanh`

The hyperbolic tangent (tanh) activation function. The forward pass is calculated as `(e^x - e^(-x))/(e^x + e^(-x))`. The backward pass is calculated as `d_output * (1 - y^2)`.

```
struct activation_tanh {
    // The input and output size.
    int input_size, output_size;

    // The input and output matrices.
    struct matrix *input, *output;

    // Gradients on the outputs and inputs.
    struct matrix *d_outputs, *d_inputs;
};
```

### `int activation_tanh_init(struct activation_tanh *obj, int input_size, struct matrix *input, struct matrix *output, struct matrix *d_outputs, struct matrix *d_inputs)`

Initialize an empty tanh layer object. Returns `1` if successful, otherwise it returns `0`.

### `void activation_tanh_forward(struct activation_tanh *obj)`

Perform a forward pass on the tanh layer.

### `void activation_tanh_backward(struct activation_tanh *obj)`

Perform a backward pass on the tanh layer.

## `layer_normalization`

The batch normalization layer, with running mean and variance, along with affine transformation (Ioffe & Szegedy, 2015).

```
struct layer_normalization {
    // The input and output dimensions.
    int input_size, output_size;
    
    // The hyperparameter values for the normalization layer.
    double epsilon, momentum;

    // Gamma and beta, the parameters to be learned.
    struct matrix gamma, beta;

    // The running mean and running variance.
    struct matrix running_mean, running_variance;
    struct matrix mean, variance;

    // The input and output matrices.
    struct matrix *input, *output;
    
    // Gradients on the outputs and inputs, respectively.
    struct matrix *d_outputs, *d_inputs;

    // Gradients on gamma and beta.
    struct matrix d_gamma, d_beta;
};
```

### `int layer_normalization_init(struct layer_normalization *obj, int input_size, double epsilon, double momentum, struct matrix *input, struct matrix *output, struct matrix *d_outputs, struct matrix *d_inputs)`

Initialize an empty batch normalization layer object. Returns `1` if successful, otherwise it returns `0`.

### `void layer_normalization_free(struct layer_normalization *obj)`

Free the matrices owned by the batch normalization layer. 

### `int layer_normalization_init_values(struct layer_normalization *obj, enum weight_initializer gamma_type, enum bias_initializer beta_type)`

Initialize gamma and beta. Returns `1` if successful, otherwise it returns `0`.

### `void layer_normalization_set_values(struct layer_normalization *obj, double epsilon, double momentum)`

Set the layer's hyper parameters.

### `void layer_normalization_forward(struct layer_normalization *obj)`

Perform a forward pass on the layer.

### `void layer_normalization_forward_predict(struct layer_normalization *obj)`

Perform a forward pass on the layer, in prediction mode.

### `void layer_normalization_backward(struct layer_normalization *obj)`

Perform a backward pass on the layer.