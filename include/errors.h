// errors.h
// Error handling.

#ifndef ERRORS_H
#define ERRORS_H

#include "declspec.h"

extern char *LAST_ERROR;

// Print the last error to stdout.
extern TOM_API void print_last_error(void);

// Return the last error.
extern TOM_API char *get_last_error(void);

// Macro to quit on error.
#define QUIT_ON_ERROR(x) { \
    int ret = (x); \
    if (!ret) { \
        print_last_error(); \
        exit(1); \
    } \
}

#endif
