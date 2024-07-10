/**
 * Instructions:
 * i) Creates an array of n random float numbers between -1.0 and 1.0. n should be read as the first command line argument as below.
 * // array -> malloc
 * ii) Scans the array using your scan function.
 * iii) Prints out the time taken by your scan function in milliseconds2.
 * iv) Prints the first element of the output scanned array.
 * v) Prints the last element of the output scanned array.
 * vi) Deallocates memory when necessary.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <chrono> // for time
#include "scan.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

#define start_t(start) start=high_resolution_clock::now(); // Get the starting timestamp
#define end_t(end) end=high_resolution_clock::now(); // Get the ending timestamp
#define duration_t(d, s, e) d=std::chrono::duration_cast<duration<double, std::milli>>(e - s);

// #define check
void print_arr (float *arr, int n) {
	for (int i = 0; i < n; i++)
		printf("%f, ", arr[i]);
	printf("]\n");
}

void init (float *arr, int n) {
	for (int i = 0; i < n; i++)
		arr[i] = (((float)rand()/RAND_MAX)*2)-1; // range [-1, 1]
}

int main(int argc, char* argv[]){
	// readin, ask memory
	int n = atoi(argv[1]);
 	float *arr    = (float*)malloc(n*sizeof(float)); // input
	float *output = (float*)malloc(n*sizeof(float)); // output
	memset(arr,    0, n*sizeof(float));
	memset(output, 0, n*sizeof(float));
	
	// time calculation
    high_resolution_clock::time_point start, end;
    duration<double, std::milli> duration_sec;

	// init
	srand(time(NULL));
	init(arr, n);
#ifdef check
	printf("pre = ["); print_arr(arr, n);
#endif

	// scan
	start_t(start)
	scan(arr, output, n);
	end_t(end)
	duration_t(duration_sec, start, end)

#ifdef check	
	printf("post = ["); print_arr(output, n);
#endif

	// printf
	printf("%lf\n", duration_sec.count()); // unit: ms
	printf("%lf\n", output[0]); // first element of the output scanned array
	printf("%lf\n", output[n-1]); // last element of the output scanned array.

	// free
	free (arr);
	free (output);
	return 0;
}
