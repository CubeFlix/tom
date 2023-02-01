// model_test.c

// segfault handling
#include <signal.h>

#include <stdlib.h>
#include <stdio.h>

#include "headers.h"

void segvHandler(int s) {
    printf("Segmentation Fault %d\n", s);
    exit(1);
}

void main() {
    signal(SIGSEGV, segvHandler);
    random_init();

    struct model *m = calloc(1, sizeof(struct model));
    model_init(m, 5);
    model_add_layer(m, LAYER_DENSE, 5, 10);
    model_add_layer(m, LAYER_DENSE, 10, 3);
    model_add_layer(m, LAYER_SOFTMAX, 3, 3);
    model_set_loss(m, LOSS_MSE);
    model_finalize(m);
    printf("finalized\n");
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