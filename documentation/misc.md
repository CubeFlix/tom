# Miscellaneous Functions

## Version

### `const char* tom_version(void)`

Returns the current `tom` version.

## Serialization

### `int serialize_matrix(struct matrix* obj, FILE* fp)`

Serialize a matrix's data to a file. Returns `1` if successful, otherwise it returns `0`.

### `int deserialize_matrix(struct matrix* obj, FILE* fp)`

Deserialize a matrix's data from a file. Returns `1` if successful, otherwise it returns `0`.

### `int serialize_layer(struct layer* obj, FILE* fp)`

Serialize a layer. Returns `1` if successful, otherwise it returns `0`.

### `int serialize_layer_params(struct layer* obj, FILE* fp)`

Serialize a layer's parameters. Returns `1` if successful, otherwise it returns `0`.

### `int deserialize_layer(struct model* obj, FILE* fp)`

Deserialize a layer and add it to a model. Returns `1` if successful, otherwise it returns `0`.

### `int deserialize_layer_params(struct layer* obj, FILE* fp)`

Deserialize a layer's parameters. Returns `1` if successful, otherwise it returns `0`.

### `int serialize_model(struct model* obj, FILE* fp)`

Serialize a model. We serialize in two passes, once for layer information, and again for layer parameters. Returns `1` if successful, otherwise it returns `0`.

### `int deserialize_model(struct model* obj, FILE* fp)`

Deserialize a model. Again, deserialize in two passes, loading layer data, initializing and finalizing the model, and then loading layer parameters. Returns `1` if successful, otherwise it returns `0`.

## Random

### `void random_init(void)`

Initialize the RNG.

### `double random_uniform(double min, double range)`

Generate a uniform random value from min to min+range.

### `double random_normal(double mu, double sigma)`

Generate a normal random value.

## Error Handling

### `LAST_ERROR`

The last error message string.

```
extern char *LAST_ERROR;
```

### `void print_last_error(void)`

Print the last error to stdout.

### `QUIT_ON_ERROR(x)`

Macro to quit on error.

```
#define QUIT_ON_ERROR(x) { \
    int ret = (x); \
    if (!ret) { \
        print_last_error(); \
        exit(1); \
    } \
}
```
