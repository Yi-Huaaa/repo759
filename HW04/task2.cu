#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "stencil.cuh"
// #define check

void init (float *image, float *mask, int n, int R) {
	int _2R_1 = 2*R+1;

    for (int i = 0; i < n; i++){
		image[i] = (((float)rand()/RAND_MAX)*2)-1; // range [-1, 1];
	}
	for (int i = 0; i < _2R_1; i++){
		mask[i] = (((float)rand()/RAND_MAX)*2)-1; // range [-1, 1];
	}
}

void init_check (float *image, float *mask, int n, int R) {
	int _2R_1 = 2*R+1;

    for (int i = 0; i < n; i++){
		image[i] = 2;
	}
	for (int i = 0; i < _2R_1; i++){
		mask[i] = 1;
	}
}


int main(int argc, char* argv[]){
	int n = atoi(argv[1]);
    int R = atoi(argv[2]);
    unsigned int  threads_per_block = atoi(argv[3]);
    int _2R_1 = 2*R+1;

    float *image  = (float*)malloc(n*sizeof(float)); 
    float *mask   = (float*)malloc((_2R_1)*sizeof(float)); 
    float *output = (float*)malloc(n*sizeof(float)); memset(output, 0, n*sizeof(float));
    float *image_gpu;   cudaMalloc((void**)&image_gpu, n*sizeof(float));
    float *mask_gpu;   cudaMalloc((void**)&mask_gpu, (_2R_1)*sizeof(float));
    float *output_gpu; cudaMalloc((void**)&output_gpu, n*sizeof(float));
    
    srand(time(NULL));
#ifdef check
    init_check(image, mask, n, R);
#else 
    init(image, mask, n, R);
#endif
    

    cudaMemcpy(image_gpu,  image,  n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mask_gpu,   mask,   _2R_1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_gpu, output, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;
    float ms; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
        
        stencil(image_gpu, mask_gpu, output_gpu, n, R, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(output, output_gpu, n*sizeof(float), cudaMemcpyDeviceToHost);


    // TODO: change to .2
#ifdef check 
    printf("%lf\n", output[n-1]); // Print the last element of the resulting matrix.
    printf("%lf\n", ms); // Print the time taken to execute your matmul function in milliseconds using CUDA events.
#else 
    printf("%.2lf\n", output[n-1]); // Print the last element of the resulting matrix.
    printf("%.2lf\n", ms); // Print the time taken to execute your matmul function in milliseconds using CUDA events.
#endif 

    free (image);
    free (mask);
    free (output);
    cudaFree(image_gpu);
    cudaFree(mask_gpu);
    cudaFree(output_gpu);

    return 0;
}