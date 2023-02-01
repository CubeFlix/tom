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

void abrtHandler(int s) {
    printf("Aborted %d\n", s);
    exit(1);
}

void main() {
    signal(SIGSEGV, segvHandler);
    signal(SIGABRT, abrtHandler);
    random_init();

    struct model *m = calloc(1, sizeof(struct model));
    model_init(m, 5);
    struct layer *l1 = model_add_layer(m, LAYER_DENSE, 5, 10);
    struct layer* l2= model_add_layer(m, LAYER_DENSE, 10, 3);
    struct layer* l3 = model_add_layer(m, LAYER_SOFTMAX, 3, 3);
    model_set_loss(m, LOSS_CROSSENTROPY);
    if (!model_finalize(m)) {
        printf(LAST_ERROR);
        exit(1);
    }
    layer_dense_init_values(l1->obj, WI_GLOROT_NORMAL, BI_ZEROS);
    printf("finalized\n");
    if (!model_init_optimizers(m, OPTIMIZER_SGD, 0.001, 0.0, 0.0)) {
        printf(LAST_ERROR);
        exit(1);
    }
    struct layer *current = m->first;
    do {
        printf("layer %d %d\n", current->input_size, current->output_size);
        fflush(stdout);
        current = current->next;
    } while (current != NULL);
    printf("loss %d %d\n", m->loss.type, m->loss.input->n_cols);
    model_free(m);
    free(m);
    printf("freed\n");
}