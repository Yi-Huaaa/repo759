#include "msort.h"

#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <cstring>

void merge(int* arr, const std::size_t n, int *out) {
    size_t i = 0, j = n / 2, k = 0;
    while (i < n / 2 && j < n) {
        if (arr[i] < arr[j]) {
            out[k++] = arr[i++];
        } else {
            out[k++] = arr[j++];
        }
    }
    while (i < n / 2) {
        out[k++] = arr[i++];
    }
    while (j < n) {
        out[k++] = arr[j++];
    }
    memcpy(arr, out, n * sizeof(int));
}

void msort_inner(int* arr, const std::size_t n, const std::size_t threshold, int* out) {
#ifdef check
    printf("%d\n", omp_get_thread_num()); // debug
#endif

    if (n < threshold || threshold == 1) {
        if (n == 1) {
            return;
        }
        // bubble sort
        for (size_t i = 0; i < n - 1; ++i) {
            for (size_t j = 0; j < n - 1 - i; ++j) {
                if (arr[j] > arr[j + 1]) {
                    int tmp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = tmp;
                }
            }
        }
        return;
    }

    #pragma omp task
    msort_inner(arr, n / 2, threshold, out);

    #pragma omp task
    msort_inner(arr + n / 2, n - n / 2, threshold, out + n / 2);

    #pragma omp taskwait

    merge(arr, n, out);
}

void msort(int* arr, const std::size_t n, const std::size_t threshold) {
 	int *out = (int *) malloc(n * sizeof(int)); // out array for merge
    #pragma omp parallel
    {
        #pragma omp single
        msort_inner(arr, n, threshold, out);
    }
    free(out);
}
