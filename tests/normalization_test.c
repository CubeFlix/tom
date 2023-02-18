// normalization_test.c

#include <stdio.h>

#include "tom.h"

int main(void) {
	struct layer_normalization l;
	struct matrix in, out;
	struct matrix d_in, d_out;
	int size = 5;
	int n_samples = 12;

	matrix_init(&in, n_samples, size);
	matrix_init(&out, n_samples, size);
	matrix_init(&d_in, n_samples, size);
	matrix_init(&d_out, n_samples, size);

	QUIT_ON_ERROR(layer_normalization_init(&l, size,  \
			PADDING_SYMMETRIC, &in, &out, &d_out, &d_in));
	
	for (int i = 0; i < in.size; i++) {
		in.buffer[i] = (double)i;
	}
	for (int i = 0; i < d_out.size; i++) {
		d_out.buffer[i] = (double)i;
	}

	printf("hello?\n");
	fflush(stdout);

	QUIT_ON_ERROR(layer_padding2d_forward(&l));
	layer_padding2d_backward(&l);
	for (int i = 0; i < l.output_height; i++) {
		for (int j = 0; j < l.output_width; j++) {
			printf("%f ", out.buffer[i * l.output_width + j]);
		}
		printf("\n");
	}
	for (int i = 0; i < l.input_height; i++) {
		for (int j = 0; j < l.input_width; j++) {
			printf("%f ", d_in.buffer[i * l.input_width + j]);
		}
		printf("\n");
	}

	layer_padding2d_free(&l);

	matrix_free(&in);
	matrix_free(&out);
	matrix_free(&d_in);
	matrix_free(&d_out);
}
