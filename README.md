# tom, a neural network library

# FEATURE BUILD (batch normalization)

> this feature build includes batch normalization layers (`layer_normalization`)

`tom` is a CPU-based neural network library written in C. It supports dense and convolutional layers, along with a multitude of different activations, optimizers, and loss functions. Internally, `tom` uses shared matrices in memory to process large amounts of double-precision floating-point data, by sharing data between layers in models. Models can also be serialized and loaded from disk. `tom` has been tested with the Iris (Fisher, 1936) and MNIST (Deng, L., 2012) datasets. See `tests/iris_test.c` and `tests/mnist_test.c` for more information.

Currently, `tom` supports the following layer types:

- Dense (fully-connected) layers ([`layer_dense`](documentation/layers.md#layer_dense))
- Convolutional 2D layers ([`layer_conv2d`](documentation/layers.md#layer_conv2d))
- Max-pooling 2D layers ([`layer_maxpool2d`](documentation/layers.md#layer_maxpool2d))
- Padding 2D layers ([`layer_padding2d.c`](documentation/layers.md#layer_padding2d))
- Dropout layers ([`layer_dropout`](documentation/layers.md#layer_dropout))
- ReLU and Leaky ReLU activation layers ([`layer_relu`](documentation/layers.md#activation_relu), [`layer_leaky_relu`](documentation/layers.md#activation_leaky_relu))
- Softmax activation layers ([`layer_softmax`](documentation/layers.md#activation_softmax))
- Sigmoid activation layers ([`layer_sigmoid`](documentation/layers.md#activation_sigmoid))
- Hyperbolic tangent (tanh) activation layers ([`layer_tanh`](documentation/layers.md#activation_tanh))

`tom` also supports the following optimizers:

- Stochastic Gradient Descent (SGD) optimizer, including Nesterov momentum ([`optimizer_sgd`](documentation/optimizers.md#optimizer_sgd), [`optimizer_sgd_conv2d`](documentation/optimizers.md#optimizer_sgd_conv2d))
- Adam optimizer (Kingma & Ba, 2014) ([`optimizer_adam`](documentation/optimizers.md#optimizer_adam), [`optimizer_adam_conv2d`](documentation/optimizers.md#optimizer_adam_conv2d))
- RMSProp optimizer (Hinton, no paper) ([`optimizer_rmsprop`](documentation/optimizers.md#optimizer_rmsprop), [`optimizer_rmsprop_conv2d`](documentation/optimizers.md#optimizer_rmsprop_conv2d))

Finally, `tom` supports the following loss functions:

- Mean Squared Error (MSE) loss ([`loss_mse`](documentation/loss.md#loss_mse))
- Mean Absolute Error (MAE) loss ([`loss_mae`](documentation/loss.md#loss_mae))
- Cross-entropy loss ([`loss_crossentropy`](documentation/loss.md#loss_crossentropy))
- Binary cross-entropy loss ([`loss_binary_crossentropy`](documentation/loss.md#loss_binary_crossentropy))

## Usage

`tom` supports CMake, and can be built using:

```
cmake --build .
```

You can use `tom` in your code as follows:

```
#include "tom.h"

int main(void) {
    struct matrix m;
    QUIT_ON_ERROR(matrix_init(&m, 5, 10));
    matrix_free(&m);

    return 0;
}
```

## Documentation

- [Examples](documentation/examples.md)
- [Model](documentation/model.md)
- [Matrix](documentation/matrix.md)
- [Layers](documentation/layers.md)
- [Loss](documentation/loss.md)
- [Optimizers](documentation/optimizers.md)
- [Misc Functions](documentation/misc.md)
