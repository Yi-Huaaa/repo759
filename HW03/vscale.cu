#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "vscale.cuh"

__global__ void vscale (const float *a, float *b, unsigned int n) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    b[idx] *= a[idx];
}