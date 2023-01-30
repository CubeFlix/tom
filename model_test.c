// model_test.c

#include <stdlib.h>
#include <stdio.h>

#include "headers.h"

void main() {
    struct model *m = calloc(1, sizeof(struct model));
    model_init(m, 5);
    model_add_layer(m, LAYER_DENSE, 5, 10);
    model_add_layer(m, LAYER_DENSE, 10, 3);
    model_add_layer(m, LAYER_DENSE, 3, 6);
    model_finalize(m);
    printf("finalized\n");
    fflush(stdout);
    struct layer *current = m->first;
    // printf("%p %p\n", (void *)current, (void *)m.first);
    // printf("layer 1: %d %p %p\n", current->input_size, current, current->next);
    // printf("layer 2: %d\n", current->next->input_size);
    do {
        printf("layer %d %d\n", current->input_size, current->output_size);
        fflush(stdout);
        current = current->next;
    } while (current != NULL);
    model_free(m);
    free(m);
    printf("freed\n");
}