#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <chrono> // for time
#include <vector>
#include "montecarlo.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

#define start_t(start) start=high_resolution_clock::now(); // Get the starting timestamp
#define end_t(end) end=high_resolution_clock::now(); // Get the ending timestamp
#define duration_t(d,s,e) d=std::chrono::duration_cast<duration<double, std::milli>>(e-s);

void init(float *x, float *y, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		x[i] = ((float) rand() / RAND_MAX) * 2 - 1; // range [-1, 1];
		y[i] = ((float) rand() / RAND_MAX) * 2 - 1; // range [-1, 1];
	}
}

void print_arr(float *arr, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		printf("%.6lf, ", arr[i]);
	}
	printf("]\n");
}

int main(int argc, char* argv[]) {
    float r = 1.0;
    size_t n = atoi(argv[1]);
    size_t t = atoi(argv[2]);
 	float *x = (float*) malloc(n * sizeof(float));
 	float *y = (float*) malloc(n * sizeof(float));
	
	// time calculation
    high_resolution_clock::time_point start, end;
    duration<double, std::milli> d2; // duration

	// init
	srand(time(NULL));
	init(x, y, n);

#ifdef check
	printf("x = ["); print_arr(x, n);
	printf("y = ["); print_arr(y, n);
#endif

	omp_set_num_threads(t);

	int incircle;
	int count = 1;
	float total_duration = 0.0;
	for (int iter = 0; iter < count; ++iter) {
		start_t(start)
		incircle = montecarlo(n, x, y, r);
		end_t(end)	
		duration_t(d2, start, end)
		total_duration += d2.count();
	}
	
	float pi = 4.0 * r * r * incircle / n;

	// print
	printf("%lf\n", pi);
    printf("%lf\n", total_duration / count); // unit: ms, d2

	free (x);
	free (y);

	return 0;
}