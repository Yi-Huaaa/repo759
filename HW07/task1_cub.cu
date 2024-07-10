#define CUB_STDERR // print CUDA runtime errors to console
#include <stdio.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "cub/util_debug.cuh"
using namespace cub;
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

// #define check

int main(int argc, char* argv[]){
    int n = atoi(argv[1]);
    
    // Set up host arrays
    float h_in[n];
    for (auto i = 0; i < n; i++) {
        h_in[i] = (((float)rand()/RAND_MAX)*2)-1; // range [-1, 1];
    }

#ifdef check
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += h_in[i];
#endif

    // Set up device arrays
    float* d_in = NULL; 
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_in, sizeof(float) * n));
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice)); // Initialize device input
    
    // Setup device output array
    float* d_sum = NULL; CubDebugExit(g_allocator.DeviceAllocate((void**)& d_sum, sizeof(float) * 1)); 

    // Request and allocate temporary storage
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));


    cudaEvent_t start;
    cudaEvent_t stop;
    float ms; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

        // Do the actual reduce operation, with removing the debug function wrapper when gauging the performance of the reduction.
        DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n);
    
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, start, stop);

    
    float gpu_sum;
    CubDebugExit(cudaMemcpy(&gpu_sum, d_sum, sizeof(float) * 1, cudaMemcpyDeviceToHost));
    // Check for correctness
#ifdef check
    printf("\t%s\n", (abs(gpu_sum - sum) < 0.01 ? "Test passed." : "Test falied."));
    printf("\tSum is: %.3lf, %.3lf\n", gpu_sum, sum);
#else 
    printf("%.3lf\n", gpu_sum);
    printf("%.3lf\n", ms);
#endif

    // Cleanup
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_sum) CubDebugExit(g_allocator.DeviceFree(d_sum));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    
    return 0;
}