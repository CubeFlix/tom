// iris_model_test.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "headers.h"

#define IRIS_SETOSA "Iris-setosa\n"
#define IRIS_VERSICOLOR "Iris-versicolor\n"
#define IRIS_VIRGINICA "Iris-virginica\n"

// error handling macro
#define QUIT_ON_ERROR(x) { \
    int ret = (x); \
    if (!ret) { \
        printf("%s\n", LAST_ERROR); \
        exit(1); \
    } \
}

static void shuffle(void *X, void *Y, size_t n, size_t size_x, size_t size_y) {
    char tmp_x[size_x];
    char tmp_y[size_y];
    size_t stride_x = size_x * sizeof(char), stride_y = size_y * sizeof(char);

    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; ++i) {
            size_t rnd = (size_t) rand();
            size_t j = i + rnd / (RAND_MAX / (n - i) + 1);

            memcpy(tmp_x, X + j * stride_x, size_x);
            memcpy(X + j * stride_x, X + i * stride_x, size_x);
            memcpy(X + i * stride_x, tmp_x, size_x);

            memcpy(tmp_y, Y + j * stride_y, size_y);
            memcpy(Y + j * stride_y, Y + i * stride_y, size_y);
            memcpy(Y + i * stride_y, tmp_y, size_y);
        }
    }
}

void load_dataset(struct matrix *X, struct matrix *Y) {
    FILE* stream = fopen("iris.csv", "r");
    char *token;
    double value;
    int i = 0;
    
    char line[1024];

    while (fgets(line, 1024, stream)) {
        // Duplicating the line, as strtok messes with tmp.
        char* tmp = strdup(line);

        token = strtok(tmp, ",");
        value = strtod(token, NULL);
        X->buffer[i * 4] = value;

        token = strtok(NULL, ",");
        value = strtod(token, NULL);
        X->buffer[i * 4 + 1] = value;

        token = strtok(NULL, ",");
        value = strtod(token, NULL);
        X->buffer[i * 4 + 2] = value;

        token = strtok(NULL, ",");
        value = strtod(token, NULL);
        X->buffer[i * 4 + 3] = value;

        token = strtok(NULL, ",");
        if (strcmp(token, IRIS_SETOSA) == 0) {
            Y->buffer[i * 3] = 1.0;
            Y->buffer[i * 3 + 1] = 0.0;
            Y->buffer[i * 3 + 2] = 0.0;
        } else if (strcmp(token, IRIS_VERSICOLOR) == 0) {
            Y->buffer[i * 3] = 0.0;
            Y->buffer[i * 3 + 1] = 1.0;
            Y->buffer[i * 3 + 2] = 0.0;
        } else {
            Y->buffer[i * 3] = 0.0;
            Y->buffer[i * 3 + 1] = 0.0;
            Y->buffer[i * 3 + 2] = 1.0;
        }

        free(tmp);
        i++;
    }
    fclose(stream);
}

int main(void) {
    // Initialize RNG.
    random_init();

    // Initialize the network and its matrices.
    int data_size = 150;
    int batch_size = 50;
    int input_size = 4;
    int h1_size = 16;
    int h2_size = 16;
    int h3_size = 3;
    
    // Prepare the training data.
    struct matrix X, Y;
    matrix_init(&X, data_size, input_size);
    matrix_init(&Y, data_size, h3_size);
    load_dataset(&X, &Y);
    shuffle(X.buffer, Y.buffer, data_size, sizeof(double) * input_size, sizeof(double) * h3_size);

    // Create the model.
    struct model* m = calloc(1, sizeof(struct model));
    model_init(m, batch_size);
    struct layer* l1 = model_add_layer(m, LAYER_DENSE, input_size, h1_size);
    struct layer* a1 = model_add_layer(m, LAYER_RELU, h1_size, h1_size);
    struct layer* l2 = model_add_layer(m, LAYER_DENSE, h1_size, h2_size);
    struct layer* a2 = model_add_layer(m, LAYER_RELU, h2_size, h2_size);
    struct layer* l3 = model_add_layer(m, LAYER_DENSE, h2_size, h3_size);
    struct layer* a3 = model_add_layer(m, LAYER_SOFTMAX, h3_size, h3_size);
    model_set_loss(m, LOSS_CROSSENTROPY);
    QUIT_ON_ERROR(model_finalize(m));
    layer_dense_init_values(l1->obj, WI_GLOROT_NORMAL, BI_ZEROS);
    layer_dense_init_values(l2->obj, WI_GLOROT_NORMAL, BI_ZEROS);
    layer_dense_init_values(l3->obj, WI_GLOROT_NORMAL, BI_ZEROS);
    QUIT_ON_ERROR(model_init_optimizers(m, OPTIMIZER_ADAM, 0.001, 0.9, 0.999, 0.0, 1e-7));

    // Train the model.
    QUIT_ON_ERROR(model_train(m, &X, &Y, 500, false));
    double loss = model_calc_loss(m, &X, &Y);
    printf("loss: %f\n", loss);

    // Model output values.
    struct matrix yhat;
    QUIT_ON_ERROR(matrix_init(&yhat, data_size, h3_size));
    QUIT_ON_ERROR(model_predict(m, &X, &yhat));
    for (int i = 0; i < data_size; i++) {
        printf("y: %f %f %f y-hat: %f %f %f\n", Y.buffer[i * h3_size], Y.buffer[i * h3_size + 1], Y.buffer[i * h3_size + 2], \
            yhat.buffer[i * h3_size], yhat.buffer[i * h3_size + 1], yhat.buffer[i * h3_size + 2]);
    }

    model_free(m);
    free(m);
    matrix_free(&X);
    matrix_free(&Y);
    matrix_free(&yhat);
    printf("done\n");
    return 0;
}