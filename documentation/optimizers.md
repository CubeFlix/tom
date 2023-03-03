# Optimizers

## `optimizer_type`

The optimizer type.

```
enum optimizer_type {
    // Stochastic gradient descent.
    OPTIMIZER_SGD,

    // Adam optimizer.
    OPTIMIZER_ADAM,

    // RMSProp optimizer.
    OPTIMIZER_RMSPROP
};
```

## `optimizer`

The generic optimizer object.

```
struct optimizer {
    // The optimizer type.
    enum optimizer_type type;

    // The current iteration.
    int iter;

    // The optimizer object.
    void* obj;
};
```

## `optimizer_sgd`

The Stochastic Gradient Descent optimizer algorithm, with momentum and decay.

```
struct optimizer_sgd {
    // The dense layer to optimize.
    struct layer_dense *layer;

    // The optimizer parameters.
    double learning_rate, momentum, decay;

    // Should we use Nesterov momentum.
    bool nesterov; 

    // The momentum matrices.
    struct matrix weight_m, bias_m;
};
```

### `int optimizer_sgd_init(struct optimizer_sgd *obj, struct layer_dense *layer, double learning_rate, double momentum, double decay, bool nesterov)`

Initialize an empty SGD optimizer object. Returns `1` if successful, otherwise it returns `0`.

### `void optimizer_sgd_free(struct optimizer_sgd *obj)`

Free the matrices owned by the SGD optimizer.

### `void optimizer_sgd_update(struct optimizer_sgd *obj, int iter)`

Update the layer's weights and biases with the SGD optimizer. `iter` should be the current optimizer iteration (for decay).

## `optimizer_sgd_conv2d`

The Stochastic Gradient Descent optimizer algorithm for conv 2D layers, with momentum and decay. 

```
struct optimizer_sgd_conv2d {
    // The conv 2D layer to optimize.
    struct layer_conv2d *layer;

    // The optimizer parameters.
    double learning_rate, momentum, decay;

	// Should we use Nesterov momentum.
	bool nesterov;

    // The momentum matrices.
    struct matrix weight_m, bias_m;
};
```

### `int optimizer_sgd_conv2d_init(struct optimizer_sgd_conv2d *obj, struct layer_conv2d *layer, double learning_rate, double momentum, double decay, bool nesterov)`

Initialize an empty SGD optimizer object. Returns `1` if successful, otherwise it returns `0`.

### `void optimizer_sgd_conv2d_free(struct optimizer_sgd_conv2d *obj)`

Free the matrices owned by the SGD optimizer.

### `void optimizer_sgd_conv2d_update(struct optimizer_sgd_conv2d *obj, int iter)`

Update the layer's weights and biases with the SGD optimizer. `iter` should be the current optimizer iteration (for decay).

## `optimizer_sgd_bn`

The Stochastic Gradient Descent optimizer algorithm for batch normalization layers, with momentum and decay. 

```
struct optimizer_sgd_bn {
    // The batch normalization layer to optimize.
    struct layer_normalization *layer;

    // The optimizer parameters.
    double learning_rate, momentum, decay;

	// Should we use Nesterov momentum.
	bool nesterov;

    // The momentum matrices.
    struct matrix gamma_m, beta_m;
};
```

### `int optimizer_sgd_bn_init(struct optimizer_sgd_bn *obj, struct layer_normalization *layer, double learning_rate, double momentum, double decay, bool nesterov)`

Initialize an empty SGD optimizer object. Returns `1` if successful, otherwise it returns `0`.

### `void optimizer_sgd_bn_free(struct optimizer_sgd_bn *obj)`

Free the matrices owned by the SGD optimizer.

### `void optimizer_sgd_bn_update(struct optimizer_sgd_bn *obj, int iter)`

Update the layer's weights and biases with the SGD optimizer. `iter` should be the current optimizer iteration (for decay).

## `optimizer_adam`

The Adam (Kingma & Ba, 2014) (adaptive moment estimation) optimizer algorithm.

```
struct optimizer_adam {
    // The dense layer to optimize.
    struct layer_dense *layer;

    // The optimizer parameters.
    double learning_rate, beta_1, beta_2, decay, epsilon;

    // The momentum matrices.
    struct matrix weight_m, bias_m, weight_c, bias_c;
};
```

### `int optimizer_adam_init(struct optimizer_adam *obj, struct layer_dense *layer, double learning_rate, double beta_1, double beta_2, double decay, double epsilon)`

Initialize an empty Adam optimizer object. Returns `1` if successful, otherwise it returns `0`.

### `void optimizer_adam_free(struct optimizer_adam *obj)`

Free the matrices owned by the Adam optimizer.

### `void optimizer_adam_update(struct optimizer_adam *obj, int iter)`

Update the layer's weights and biases with the Adam optimizer. `iter` should be the current optimizer iteration (for decay).

## `optimizer_adam_conv2d`

The Adam (Kingma & Ba, 2014) (adaptive moment estimation) optimizer algorithm for conv 2D layers.

```
struct optimizer_adam_conv2d {
    // The conv 2D layer to optimize.
    struct layer_conv2d *layer;

    // The optimizer parameters.
    double learning_rate, beta_1, beta_2, decay, epsilon;

    // The momentum matrices.
    struct matrix weight_m, bias_m, weight_c, bias_c;
};
```

### `int optimizer_adam_conv2d_init(struct optimizer_adam_conv2d *obj, struct layer_conv2d *layer, double learning_rate, double beta_1, double beta_2, double decay, double epsilon)`

Initialize an empty Adam optimizer object. Returns `1` if successful, otherwise it returns `0`.

### `void optimizer_adam_conv2d_free(struct optimizer_adam_conv2d *obj)`

Free the matrices owned by the Adam optimizer.

### `void optimizer_adam_conv2d_update(struct optimizer_adam_conv2d *obj, int iter)`

Update the layer's weights and biases with the Adam optimizer. `iter` should be the current optimizer iteration (for decay).

## `optimizer_adam_bn`

The Adam (Kingma & Ba, 2014) (adaptive moment estimation) optimizer algorithm for batch normalization layers.

```
struct optimizer_adam_bn {
    // The normalization layer to optimize.
    struct layer_normalization *layer;

    // The optimizer parameters.
    double learning_rate, beta_1, beta_2, decay, epsilon;

    // The momentum matrices.
    struct matrix gamma_m, beta_m, gamma_c, beta_c;
};
```

### `int optimizer_adam_bn_init(struct optimizer_adam_bn *obj, struct layer_normalization *layer, double learning_rate, double beta_1, double beta_2, double decay, double epsilon)`

Initialize an empty Adam optimizer object. Returns `1` if successful, otherwise it returns `0`.

### `void optimizer_adam_bn_free(struct optimizer_adam_bn *obj)`

Free the matrices owned by the Adam optimizer.

### `void optimizer_adam_bn_update(struct optimizer_adam_bn *obj, int iter)`

Update the layer's weights and biases with the Adam optimizer. `iter` should be the current optimizer iteration (for decay).

## `optimizer_rmsprop`

The Root Mean Square Propagation optimizer algorithm.

```
struct optimizer_rmsprop {
    // The dense layer to optimize.
    struct layer_dense *layer;

    // The optimizer parameters.
    double learning_rate, decay, epsilon, rho;

    // The cache matrices.
    struct matrix weight_c, bias_c;
};
```

### `int optimizer_rmsprop_init(struct optimizer_rmsprop *obj, struct layer_dense *layer, double learning_rate, double decay, double epsilon, double rho)`

Initialize an empty RMSProp optimizer object. Returns `1` if successful, otherwise it returns `0`.

### `void optimizer_rmsprop_free(struct optimizer_rmsprop *obj)`

Free the matrices owned by the RMSProp optimizer.

### `void optimizer_rmsprop_update(struct optimizer_rmsprop *obj, int iter)`

Update the layer's weights and biases with the RMSProp optimizer. `iter` should be the current optimizer iteration (for decay).

## `optimizer_rmsprop_conv2d`

The Root Mean Square Propagation optimizer algorithm for conv 2D layers.

```
struct optimizer_rmsprop_conv2d {
    // The conv 2D layer to optimize.
    struct layer_conv2d *layer;

    // The optimizer parameters.
    double learning_rate, decay, epsilon, rho;

    // The cache matrices.
    struct matrix weight_c, bias_c;
};
```

### `int optimizer_rmsprop_conv2d_init(struct optimizer_rmsprop_conv2d *obj, struct layer_conv2d *layer, double learning_rate, double decay, double epsilon, double rho)`

Initialize an empty RMSProp optimizer object. Returns `1` if successful, otherwise it returns `0`.

### `void optimizer_rmsprop_conv2d_free(struct optimizer_rmsprop_conv2d *obj)`

Free the matrices owned by the RMSProp optimizer.

### `void optimizer_rmsprop_conv2d_update(struct optimizer_rmsprop_conv2d *obj, int iter)`

Update the layer's weights and biases with the RMSProp optimizer. `iter` should be the current optimizer iteration (for decay).

## `optimizer_rmsprop_bn`

The Root Mean Square Propagation optimizer algorithm for batch normalization layers.

```
struct optimizer_rmsprop_bn {
    // The normalization layer to optimize.
    struct layer_normalization *layer;

    // The optimizer parameters.
    double learning_rate, decay, epsilon, rho;

    // The cache matrices.
    struct matrix gamma_c, beta_c;
};
```

### `int optimizer_rmsprop_bn_init(struct optimizer_rmsprop_bn *obj, struct layer_normalization *layer, double learning_rate, double decay, double epsilon, double rho)`

Initialize an empty RMSProp optimizer object. Returns `1` if successful, otherwise it returns `0`.

### `void optimizer_rmsprop_bn_free(struct optimizer_rmsprop_bn *obj)`

Free the matrices owned by the RMSProp optimizer.

### `void optimizer_rmsprop_bn_update(struct optimizer_rmsprop_bn *obj, int iter)`

Update the layer's weights and biases with the RMSProp optimizer. `iter` should be the current optimizer iteration (for decay).

