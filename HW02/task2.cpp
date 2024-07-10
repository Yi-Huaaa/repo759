/**
 *  Instructions: 
 * i) Creates an n×n image matrix (stored in 1D in row-major order) of random float numbers between -10.0 and 10.0. The value of n should be read as the first command line argument.
 * ii) Creates an m×m mask matrix (stored in 1D in row-major order) of random float numbers between -1.0 and 1.0. The value of m should be read as the second command line argument.
 * iii) Applies the mask to image using your convolve function.
 * iv) Prints out the time taken by your convolve function in milliseconds.
 * v) Prints the first element of the resulting convolved array.
 * vi) Prints the last element of the resulting convolved array.
 * vii) Deallocates memory when necessary via the delete function. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <chrono> // for time
#include "convolution.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

#define start_t(start) start=high_resolution_clock::now(); // Get the starting timestamp
#define end_t(end) end=high_resolution_clock::now(); // Get the ending timestamp
#define duration_t(d,s,e) d=std::chrono::duration_cast<duration<double, std::milli>>(e-s);

void init_test (float *image, float *mask, int n, int m) {
	float ii [16] = {1,3,4,8,6,5,2,4,3,4,6,8,1,4,5,2};
	float mm [9] = {0,0,1,0,1,0,1,0,0};
	for (int i = 0; i < n*n; i++)
		image[i] = ii[i];
	for (int i = 0; i < m*m; i++)
		mask[i] = mm[i];
}

void init (float *image, float *mask, int n, int m) {
	for (int i = 0; i < n*n; i++)
		image[i] = (((float)rand()/RAND_MAX)*20)-10; // range [-10, 10]
	for (int i = 0; i < m*m; i++)
		mask[i] = (((float)rand()/RAND_MAX)*2)-1; // range [-1, 1]
}

// #define check
void print_arr (float *arr, int sz) {
	for (int i = 0; i < sz; i++) {
		printf("[");
		for (int j = 0; j < sz; j++) {
			printf("%f, ", arr[i*sz+j]);
		}
		printf("]\n");
	}
	printf("]\n");
}

int main(int argc, char* argv[]){
	int n = atoi(argv[1]);
	int m = atoi(argv[2]);

	// float *image  = (float*)malloc(n*n*sizeof(float));
	// float *mask   = (float*)malloc(m*m*sizeof(float)); 
	// float *output = (float*)malloc(n*n*sizeof(float));
	float* image  = new float[n*n];
	float* mask   = new float[m*m];
	float* output = new float[n*n];

	// time calculation
    high_resolution_clock::time_point start, end;
    duration<double, std::milli> d1; // duration

	// init
	srand(time(NULL));
#ifdef check
	init_test(image, mask, n, m);
#else 
	init(image, mask, n, m);
#endif

#ifdef check
	printf("image = ["); print_arr(image, n);
	printf("mask = [");  print_arr(mask, m);
#endif
	
	// convolve
	start_t(start)
	convolve(image, output, n, mask, m);
	end_t(end)	
	duration_t(d1, start, end)

	// print 
	printf("%lf\n", d1.count()); // unit: ms, d1
	printf("%lf\n", output[0]);
	printf("%lf\n", output[n*n-1]);

#ifdef check
	printf("output = ["); print_arr(output, n);
#endif

	// free (image);
	// free (mask);
	// free (output);
	delete[] image;
	delete[] mask;
	delete[] output;

	return 0;
}