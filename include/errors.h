// errors.h
// Error handling.

#ifndef ERRORS_H
#define ERRORS_H

#include "declspec.h"

extern char *LAST_ERROR;

// Print the last error to stdout.
extern TOM_API void error_print();

// Macro to quit on error.
#define QUIT_ON_ERROR(x) { \
    int ret = (x); \
    if (!ret) { \
        error_print() \
        exit(1); \
    } \
}

#endif
