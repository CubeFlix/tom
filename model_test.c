// model_test.c

#include <stdio.h>

#include "headers.h"

void main() {
    struct model m;
    model_init(&m, 5);
    model_add_layer(&m, LAYER_DENSE, 5, 10);
    model_add_layer(&m, LAYER_DENSE, 10, 3);
    model_finalize(&m);
    printf("finalized\n");
    fflush(stdout);
    struct layer *current = m.first;
    //do {
        //printf("layer %d %d\n", current->input_size, current->output_size);
        //fflush(stdout);
        //current = current->next;
    //} while (current != NULL);
    model_free(&m);
    printf("freed\n");
}