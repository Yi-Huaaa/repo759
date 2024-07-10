/**
 * Instructions:
 * generates square matrices A and B of dimension at least 1000×1000 stored in row-major order.
 * computes the matrix product C = AB using each of your functions (note that you may have to prepare A and B in different data types so they comply with the function argument types). Your result stored in matrix C should be the same no matter which function defined at a) through d) above you call.
 * prints the number of rows of your input matrices, and for each mmul function in ascending order, prints the amount of time taken in milliseconds and the last element of the resulting C. There should be nine values printed, one per line
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <chrono> // for time
#include <vector>
#include "matmul.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

#define start_t(start) start=high_resolution_clock::now(); // Get the starting timestamp
#define end_t(end) end=high_resolution_clock::now(); // Get the ending timestamp
#define duration_t(d,s,e) d=std::chrono::duration_cast<duration<double, std::milli>>(e-s);

void init(float *a, float *b, size_t n) {
	for (size_t i = 0; i < n*n; i++) {
#ifdef check
		a[i] = 1;
		b[i] = 1;
#else
		a[i] = (((float)rand()/RAND_MAX)*2)-1; // range [-1, 1];
		b[i] = (((float)rand()/RAND_MAX)*2)-1; // range [-1, 1];
#endif
	}
}

void print_max(float *arr, size_t n) {
	for (size_t i = 0; i < n; i++) {
		printf("[");
		for (size_t j = 0; j < n; j++) {
			printf("%.17lf, ", arr[i*n+j]);
		}
		printf("],\n");
	}
	printf("]\n");
}

int main(int argc, char* argv[]) {
    size_t n = atoi(argv[1]);
    size_t t = atoi(argv[2]);
 	float *A     = (float*)malloc(n*n*sizeof(float)); // input_A
	float *B     = (float*)malloc(n*n*sizeof(float)); // input_B
	float *C = (float*)malloc(n*n*sizeof(float)); // output_2
	
	memset(C, 0, n*n*sizeof(float));
	
	// time calculation
    high_resolution_clock::time_point start, end;
    duration<double, std::milli> d2; // duration

	// init
	srand(time(NULL));
	init(A, B, n);	

#ifdef check
	printf("A = ["); print_max(A, n);
	printf("B = ["); print_max(B, n);
#endif
	// TODO: check if put this here is correct
	omp_set_num_threads(t);

	// mmul2
	start_t(start)
	mmul(A, B, C, n);
	end_t(end)	
	duration_t(d2, start, end)

#ifdef check
	printf("C = ["); print_max(C, n);
#endif

	// print
    // NOTE: print output is different than HW02
	printf("%lf\n", C[0]); // first element of the resulting C
	printf("%lf\n", C[n*n-1]); // last element of the resulting C
    printf("%lf\n", d2.count()); // unit: ms, d2

	free (A);
	free (B);
	free (C);

	return 0;
}