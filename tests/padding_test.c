// padding_test.c

#include <stdio.h>

#include "tom.h"

int main(void) {
	struct layer_padding2d l;
	struct matrix in, out;
	struct matrix d_in, d_out;
	int width = 5, height = 5;
	int padding_x = 2, padding_y = 2;

	matrix_init(&in, 1, width * height);
	matrix_init(&out, 1, (width + padding_x * 2) * (height + padding_y * 2));
	matrix_init(&d_in, 1, width * height);
	matrix_init(&d_out, 1, (width + padding_x * 2) * (height + padding_y * 2));

	QUIT_ON_ERROR(layer_padding2d_init(&l, 1, height, width, padding_x, padding_y, \
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
