#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <chrono> // for time
#include <vector>
#include "msort.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

#define start_t(start) start=high_resolution_clock::now(); // Get the starting timestamp
#define end_t(end) end=high_resolution_clock::now(); // Get the ending timestamp
#define duration_t(d,s,e) d=std::chrono::duration_cast<duration<double, std::milli>>(e-s);

void init(int *a, size_t n) {
	for (size_t i = 0; i < n; i++) {
#ifdef check
		a[i] = n - 1 - i;
#else
		a[i] = (rand() % 2001) - 1000; // range [-1000, 1000];
#endif
	}
}

void print_max(int *arr, size_t n) {
	printf("[");
    for (size_t i = 0; i < n; i++) {
		printf("%d, ", arr[i]);
	}
	printf("]\n");
}

int main(int argc, char* argv[]) {
    size_t n = atoi(argv[1]);
    size_t t = atoi(argv[2]);
    size_t ts = atoi(argv[3]);
 	int *arr     = (int *) malloc(n * sizeof(int)); // input_arr
	
	// time calculation
    high_resolution_clock::time_point start, end;
    duration<double, std::milli> d2; // duration

	// init
	srand(time(NULL));
	init(arr, n);	

#ifdef check
	print_max(arr, n);
#endif
	// TODO: check if put this here is correct
	omp_set_num_threads(t);

	// mmul2
	start_t(start)
	msort(arr, n, ts);
	end_t(end)	
	duration_t(d2, start, end)

#ifdef check
	print_max(arr, n);
#endif

	// print
    // NOTE: print output is different than HW02
	printf("%d\n", arr[0]); // first element of the resulting C
	printf("%d\n", arr[n - 1]); // last element of the resulting C
    printf("%lf\n", d2.count()); // unit: ms, d2

	free(arr);

	return 0;
}