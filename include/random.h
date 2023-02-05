// random.c
// Random number generation.

// Initialize the RNG.
void random_init();

// Generate a uniform random value from min to min+range.
double random_uniform(double min, double range);

// Generate a normal random value.
double random_normal(double mu, double sigma);