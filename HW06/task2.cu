#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "scan.cuh"

// #define check 

void init (float *input, float *output, const unsigned int n) {
    for (unsigned int i = 0; i < n; i++){
        input[i] = (((float)rand()/RAND_MAX)*2)-1; // range [-1, 1];
        output[i] = 0;
	}
}

void init_check (float *input, float *output, const unsigned int n) {
    for (unsigned int i = 0; i < n; i++){
        input[i]  = i;
        output[i] = 0;
	}
}

void print_arr (const float *arr, int len) {
    printf("arr = [");
    for (auto i = 0; i < len; i++) {
        printf("%.2lf, ", arr[i]);
    } printf("]\n");
}


int main(int argc, char* argv[]){
	unsigned int n = atoi(argv[1]);
    unsigned int threads_per_block = atoi(argv[2]);

    float *input, *output;
    cudaMallocManaged(&input, n*sizeof(float));
    cudaMallocManaged(&output, n*sizeof(float));

    srand(time(NULL));
#ifdef check 
    init_check(input, output, n);
#else     
    init (input, output, n);
#endif 

    cudaEvent_t start;
    cudaEvent_t stop;
    float ms; 

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

        scan(input, output, n, threads_per_block);
    
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, start, stop);

    printf("%.2lf\n", output[n-1]); // Print the last element of the array containing the output of the inclusive scan operation.
    printf("%.2lf\n", ms); // Print the time taken to run the full scan function in milliseconds using CUDA events.

    
    cudaFree(input);
    cudaFree(output);

	return 0;
}
