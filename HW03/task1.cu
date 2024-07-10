#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void kernel (){
    int a = (threadIdx.x+1), b = 1;
    for (int i = 1; i < (a+1); i++)
        b *= i; 

    printf("%d!=%d\n", a, b);
}


int main(int argc, char* argv[]){

    kernel <<< 1, 8 >>> ();
    cudaDeviceSynchronize();

	return 0;
}
