#include "matmul.h"

#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>

// order: i, k, j
void mmul(const float* A, const float* B, float* C, const std::size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) {
            for (size_t j = 0; j < n; j++) {
                C[i * n + j]+= A[i * n + k] * B[k * n + j];
            }
        }
#ifdef check
        printf("%d\n", omp_get_thread_num()); // debug
#endif
    }
}
