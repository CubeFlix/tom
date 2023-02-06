// errors.h
// Error handling.

#include <stdio.h>

#include "errors.h"

char *LAST_ERROR;

// Print the last error to stdout.
void print_last_error() {
    printf("%s\n", LAST_ERROR);
}
