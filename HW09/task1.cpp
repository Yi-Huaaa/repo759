#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <chrono> // for time
#include <vector>
#include <algorithm>
#include "cluster.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

#define start_t(start) start=high_resolution_clock::now(); // Get the starting timestamp
#define end_t(end) end=high_resolution_clock::now(); // Get the ending timestamp
#define duration_t(d,s,e) d=std::chrono::duration_cast<duration<double, std::milli>>(e-s);

void init(float *arr, float *centers, size_t n, size_t t) {
	for (size_t i = 0; i < n; ++i) {
		arr[i] = (((float) rand() / RAND_MAX) * n); // range [0, n];
	}

	for (size_t i = 0; i < t; ++i) {
		centers[i] = (2 * i + 1) * n / (2 * t);
	}
}

void print_arr(float *arr, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		printf("%.6lf, ", arr[i]);
	}
	printf("]\n");
}

int main(int argc, char* argv[]) {
    size_t n = atoi(argv[1]);
    size_t t = atoi(argv[2]);
 	float *arr     = (float*) malloc(n * sizeof(float));
 	float *centers = (float*) malloc(t * sizeof(float));
 	float *dists   = (float*) malloc(t * sizeof(float));
	
	// time calculation
    high_resolution_clock::time_point start, end;
    duration<double, std::milli> d2; // duration

	// init
	srand(time(NULL));
	init(arr, centers, n, t);
	memset(dists, 0, t * sizeof(float));

	// sort
	std::sort(arr, arr + n);

#ifdef check
	printf("arr = ["); print_arr(arr, n);
	printf("centers = ["); print_arr(centers, t);
#endif
	start_t(start)
	cluster(n, t, arr, centers, dists);
	end_t(end)	
	duration_t(d2, start, end)

#ifdef check
	printf("dists = ["); print_arr(dists, t);
#endif

	float max_dist = -1.0;
	size_t who = -1;
	for (size_t i = 0; i < t; ++i) {
		if (dists[i] > max_dist) {
			max_dist = dists[i];
			who = i;
		}
	}
	// print
	printf("%lf\n", max_dist);
	printf("%ld\n", who);
    printf("%lf\n", d2.count()); // unit: ms, d2

	free (arr);
	free (centers);
	free (dists);

	return 0;
}