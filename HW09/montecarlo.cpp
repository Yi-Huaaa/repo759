#include "montecarlo.h"

int montecarlo(const size_t n, const float *x, const float *y, const float radius) {
    int incircle = 0;
#pragma omp parallel
{
    #pragma omp for simd reduction (+:incircle)
    for (size_t i = 0; i < n; ++i) {
        incircle += (x[i] * x[i] + y[i] * y[i] <= radius * radius);
    }
}
    return incircle;
}
