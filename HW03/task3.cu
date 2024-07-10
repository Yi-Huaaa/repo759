#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "vscale.cuh"

// #define check
// #define exp2


void init (float *a, float *b, int n) {
	for (int i = 0; i < n; i++){
        a[i] = (((float)rand()/RAND_MAX)*20)-10; // range [-10.0, 10.0]
        b[i] = (((float)rand()/RAND_MAX)); // range [0.0, 1.0]
    }
}

int main(int argc, char* argv[]){
	int n = atoi(argv[1]);

    float *a = (float*)malloc(n*sizeof(float));
    float *b = (float*)malloc(n*sizeof(float));
    float *da; cudaMalloc((void**)&da, n*sizeof(float));
    float *db; cudaMalloc((void**)&db, n*sizeof(float));
    
    srand(time(NULL));
    init (a, b, n);
    cudaMemcpy(da, a, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, n*sizeof(float), cudaMemcpyHostToDevice);

    int num_thd, num_blk;
    num_thd = 512, num_blk = int(n/num_thd);
#ifdef check
    printf("n = %d #num_thd = %d, num_blk = %d\n", n, num_thd, num_blk);
    printf("a = [");
    for (auto i = 0; i < n; i++) {
        printf("%3lf, ", a[i]);
    } printf("]\nb = [");
    for (auto i = 0; i < n; i++) {
        printf("%3lf, ", b[i]);
    } printf("]\n");
#endif

    cudaEvent_t start;
    cudaEvent_t stop;
    float ms; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
        
        vscale <<< num_blk, num_thd >>> (da, db, n);
        
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaDeviceSynchronize();
    cudaMemcpy(b, db, n*sizeof(float), cudaMemcpyDeviceToHost);

    // print
    printf("%3lf\n", ms);
    printf("%1lf\n", b[0]);
    printf("%1lf\n", b[n-1]);

#ifdef check
    printf("output_b = [");
    for (auto i = 0; i < n; i++) {
        printf("%3lf, ", b[i]);
    } printf("]\n");
#endif
    
#ifdef exp2
    // experiment-2 
    cudaMemcpy(db, b, n*sizeof(float), cudaMemcpyHostToDevice);
    num_thd = 16, num_blk = int(n/num_thd);
    printf("16 threads, num_thd =%d, num_blk = %d\n", num_thd, num_blk);
  
    ms = 0.0;
    cudaEventRecord(start);
    
        vscale <<< num_blk, num_thd >>> (da, db, n);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaDeviceSynchronize();
    printf("%3lf\n", ms);

#endif 
    
    free (a);
    free (b);
    cudaFree(da);
    cudaFree(db);

	return 0;
}
