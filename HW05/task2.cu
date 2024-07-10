#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "reduce.cuh"
// #define check

void init (unsigned int N, float *a) {
	for (auto i = 0; i < int(N); i++){
		a[i] = (((float)rand()/RAND_MAX)*2)-1; // range [-1, 1];
	}
}

void init_check (unsigned int N, float *a) {
	for (auto i = 0; i < int(N); i++){
		a[i] = 1;
	}
}

void print_arr (const float *arr, int len) {
    printf("arr = [");
    for (auto i = 0; i < len; i++) {
        printf("%.2lf, ", arr[i]);
    } printf("]\n");
}

unsigned int pad_exp_2_cpu (unsigned int threads_per_block) {
    int tpb = 1; 
    unsigned int exp2_arr [11] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    for (int i = 0; i < 11; i++) {
        if (exp2_arr[i] >= threads_per_block) {
            tpb = int(exp2_arr[i]);
            break;
        }
    }
    return tpb;
}

int main(int argc, char* argv[]){
	unsigned int N = atoi(argv[1]);
    unsigned int threads_per_block = atoi(argv[2]);

    unsigned int tpb = pad_exp_2_cpu(threads_per_block);
    unsigned int deal_num_per_block = 2*tpb; // since one thread processes 2 number simultaneously
    unsigned int num_block = (N > deal_num_per_block) ? (((N+deal_num_per_block-1)/deal_num_per_block)) : (1);

    float *input = (float*)malloc(N*sizeof(float));  memset(input, 0, N*sizeof(float));
    float *input_gpu; cudaMalloc((void**)&input_gpu, N*sizeof(float));
    float *output_gpu; cudaMalloc((void**)&output_gpu, num_block*sizeof(float));
    
    srand(time(NULL));
#ifdef check
    init_check(N, input);    
#else 
    init(N, input);    
#endif

    cudaMemcpy(input_gpu, input, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_gpu, input, num_block*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;
    float ms; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
        
        reduce(&input_gpu, &output_gpu, N, threads_per_block);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaMemcpy(input, input_gpu, 1*sizeof(float), cudaMemcpyDeviceToHost); 
    printf("%.2lf\n", input[0]); // Print the resulting sum.
    printf("%.2lf\n", ms); // time

    free(input);
    cudaFree(input_gpu);
    cudaFree(output_gpu);

	return 0;
}
