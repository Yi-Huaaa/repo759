#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// #define check

__global__ void kernel (int a, int *dA) {
    int x = threadIdx.x, y = blockIdx.x, idx = threadIdx.x + blockDim.x*blockIdx.x;
    dA[idx] = a*x+y;
}

int main(int argc, char* argv[]){

    int *hA = (int*)malloc(16*sizeof(int)); memset(hA, 0, 16*sizeof(int));
    int *dA; cudaMalloc((void**)&dA, 16*sizeof(int));
    cudaMemcpy(dA, hA, 16*sizeof(int), cudaMemcpyHostToDevice);

    srand(time(NULL));
    int a = rand() % ((1<<25)-1);
#ifdef check
    printf("a = %d\n", a);
#endif

    kernel <<< 2, 8 >>> (a, dA);

    cudaDeviceSynchronize();
    cudaMemcpy(hA, dA, 16*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = 0; i < 16; i++) 
        printf("%d ", hA[i]);

    free (hA);
    cudaFree (dA);

	return 0;
}
