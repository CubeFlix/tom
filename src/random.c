// random.c
// Random number generation.

#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "random.h"

// Initialize the RNG.
void random_init(void) {
    srand((unsigned int)time(NULL)); 
}

// Generate a uniform random value from min to min+range.
double random_uniform(double min, double range) {
    return (double)rand() / RAND_MAX * (range) + min;
}

// Generate a normal random value. Source: https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
double random_normal(double mu, double sigma) {
    double U1, U2, W, mult;
    static double X1, X2;
    static int call = 0;

    if (call == 1) {
        call = !call;
        return (mu + sigma * (double) X2);
    }

    do {
        U1 = -1 + ((double)rand() / RAND_MAX) * 2;
        U2 = -1 + ((double)rand() / RAND_MAX) * 2;
        W = pow(U1, 2) + pow(U2, 2);
    }
    while (W >= 1 || W == 0);

    mult = sqrt((-2 * log(W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;

    call = !call;

    return (mu + sigma * (double)X1);
}