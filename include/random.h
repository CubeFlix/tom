// random.c
// Random number generation.

#ifndef RANDOM_H
#define RANDOM_H

#include "declspec.h"

// Initialize the RNG.
extern TOM_API void random_init(void);

// Generate a uniform random value from min to min+range.
extern TOM_API double random_uniform(double min, double range);

// Generate a normal random value.
extern TOM_API double random_normal(double mu, double sigma);

#endif