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

#define N 1024 // matrix size 
#define start_t(start) start=high_resolution_clock::now(); // Get the starting timestamp
#define end_t(end) end=high_resolution_clock::now(); // Get the ending timestamp
#define duration_t(d,s,e) d=std::chrono::duration_cast<duration<double, std::milli>>(e-s);

void init (double *a, double *b, std::vector<double> &a_vec, std::vector<double> &b_vec) {
	for (int i = 0; i < N*N; i++){
		a[i] = (((double)rand()/RAND_MAX)*2)-1; // range [-1, 1];
		a_vec.push_back(a[i]);
		b[i] = (((double)rand()/RAND_MAX)*2)-1; // range [-1, 1];
		b_vec.push_back(b[i]);
	}
}

// #define check
void print_max (double *arr) {
	for (int i = 0; i < N; i++) {
		printf("[");
		for (int j = 0; j < N; j++) {
			printf("%.17lf, ", arr[i*N+j]);
		}
		printf("],\n");
	}
	printf("]\n");
}

int main(int argc, char* argv[]){
 	double *A     = (double*)malloc(N*N*sizeof(double)); // input_A
	double *B     = (double*)malloc(N*N*sizeof(double)); // input_B
	double *out_1 = (double*)malloc(N*N*sizeof(double)); // output_1
	double *out_2 = (double*)malloc(N*N*sizeof(double)); // output_2
	double *out_3 = (double*)malloc(N*N*sizeof(double)); // output_3
	double *out_4 = (double*)malloc(N*N*sizeof(double)); // output_4
	std::vector<double> A_vec, B_vec;
	
	memset(out_1, 0, N*N*sizeof(double));
	memset(out_2, 0, N*N*sizeof(double));
	memset(out_3, 0, N*N*sizeof(double));
	memset(out_4, 0, N*N*sizeof(double));
	
	// time calculation
    high_resolution_clock::time_point start, end;
    duration<double, std::milli> d1, d2, d3, d4; // duration

	// init
	srand(time(NULL));
	init(A, B, A_vec, B_vec);	

#ifdef check
	printf("A = ["); print_max(A);
	printf("B = ["); print_max(B);
#endif

	// mmul1
	start_t(start)
	mmul1(A, B, out_1, N);
	end_t(end)	
	duration_t(d1, start, end)

	// mmul2
	start_t(start)
	mmul2(A, B, out_2, N);
	end_t(end)	
	duration_t(d2, start, end)

	// mmul3
	start_t(start)
	mmul3(A, B, out_3, N);
	end_t(end)	
	duration_t(d3, start, end)

	// mmul4
	start_t(start)
	mmul4(A_vec, B_vec, out_4, N);
	end_t(end)	
	duration_t(d4, start, end)

#ifdef check
	printf("out_1 = ["); print_max(out_1);
	printf("out_2 = ["); print_max(out_2);
	printf("out_3 = ["); print_max(out_3);
	printf("out_4 = ["); print_max(out_4);
#endif

	// print
	printf("%d\n", N);
	printf("%lf\n", d1.count()); // unit: ms, d1
	printf("%lf\n", out_1[N*N-1]); //  last element of the resulting C
	printf("%lf\n", d2.count()); // unit: ms, d2
	printf("%lf\n", out_2[N*N-1]); //  last element of the resulting C
	printf("%lf\n", d3.count()); // unit: ms, d3
	printf("%lf\n", out_3[N*N-1]); //  last element of the resulting C
	printf("%lf\n", d4.count()); // unit: ms, d4
	printf("%lf\n", out_4[N*N-1]); //  last element of the resulting C

	free (A);
	free (B);
	free (out_1);
	free (out_2);
	free (out_3);
	free (out_4);
	A_vec.clear();
	B_vec.clear();

	return 0;
}