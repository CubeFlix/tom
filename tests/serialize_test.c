// serialize_test.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "headers.h"

// error handling macro
#define QUIT_ON_ERROR(x) { \
    int ret = (x); \
    if (!ret) { \
        printf("%s\n", LAST_ERROR); \
        exit(1); \
    } \
}

static void serialize(void) {
	struct model m = { 0 };
	model_init(&m, 100);
	model_add_layer(&m, LAYER_DENSE, 10, 100);
	model_add_layer(&m, LAYER_RELU, 100, 100);
	model_add_layer(&m, LAYER_DENSE, 100, 10);
	model_add_layer(&m, LAYER_SOFTMAX, 10, 10);
	model_set_loss(&m, LOSS_CROSSENTROPY);
	QUIT_ON_ERROR(model_finalize(&m));
	
	FILE* fp = fopen("modelout.dat", "wb");
	serialize_model(&m, fp);
	fclose(fp);

	model_free(&m);
}

static void deserialize(void) {
	struct model m = { 0 };
	model_init(&m, 100);
	FILE* fp = fopen("modelout.dat", "rb");
	QUIT_ON_ERROR(deserialize_model(&m, fp));
	fclose(fp);

	// print out the model
	struct layer* current = m.first;
	do {
		printf("layer in: %d out: %d\n", current->input_size, current->output_size);

		current = current->next;
	} while (current != NULL);
	
	model_free(&m);
}

int main(int argc, char *argv[]) {
	if (argc != 2) {
		printf("provide an option: \"serialize\" or \"deserialize\"\n");
		return 1;
	}
	if (!strcmp(argv[1], "serialize")) {
		printf("serialize\n");
		serialize();
	}
	else if (!strcmp(argv[1], "deserialize")) {
		printf("deserialize\n");
		deserialize();
	}
	else {
		printf("invalid option\n");
		return 1;
	}
	return 0;
}