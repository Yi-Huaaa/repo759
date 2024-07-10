#include <stdio.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// #define check

void print_arr (thrust::host_vector<float> h_vec, int n) {
    printf("h_vec = [");
    for (auto i = 0; i < n; i++) {
        printf("%.2lf, ", h_vec[i]);
    } printf("]\n");
}

int main(int argc, char* argv[]){
	int n = atoi(argv[1]);
    thrust::host_vector<float> h_vec(n);

#ifdef check 
    for (auto i = 0; i < n; i++) {
        h_vec[i] = i;
    }
    // print_arr(h_vec, n);
#else 
    for (auto i = 0; i < n; i++) {
        h_vec[i] = (((float)rand()/RAND_MAX)*2)-1; // range [-1, 1];
    }
#endif  

    // transfer data to the device
    thrust::device_vector<float> d_vec = h_vec;

    cudaEvent_t start;
    cudaEvent_t stop;
    float ms; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

        float sum = thrust::reduce(d_vec.begin(), d_vec.end()); // Call the thrust::reduce function to perform a reduction on the previously generated thrust::device vector.
    
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, start, stop);
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin()); // transfer data back to host

    printf("%.3lf\n", sum);
    printf("%.3lf\n", ms);

    // vector memory automatically released w/ free() or cudaFree()

    return 0;
}
