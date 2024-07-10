#include <stdio.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/unique.h>

#include "count.cuh"

// float operator () (int idx) {
//     thrust::default_random_engine randEng;
//     thrust::uniform_real_distribution<int> uniDist(0, 500);
//     randEng.discard(idx);
//     return uniDist(randEng);
// }


// #define check

void print_arr (thrust::host_vector<float> h_vec, int n) {
    printf("h_vec = [");
    for (auto i = 0; i < n; i++) {
        printf("%.2lf, ", h_vec[i]);
    } printf("]\n");
}

// TODO: Create and fill with random int numbers in the range [0, 500] a thrust::host vector of length n where n is the first command line argument as below.

int main(int argc, char* argv[]){
	int n = atoi(argv[1]);
    thrust::host_vector<int> h_vec(n);

#ifdef check 
    for (auto i = 0; i < n; i++) {
        h_vec[i] = i % 4;
    }
    // print_arr(h_vec, n);
#else 
    srand(time(NULL));
    for (auto i = 0; i < n; i++) {
        h_vec[i] = rand() % 501; 
    }
#endif  

    // transfer data to the device
    thrust::device_vector<int> d_in = h_vec;

    // Allocate two other thrust::device vectors, values and counts, then call your count    
    thrust::device_vector<int> values, counts;

    cudaEvent_t start;
    cudaEvent_t stop;
    float ms; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

        count(d_in, values, counts);
    
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, start, stop);
    // thrust::copy(d_in.begin(), d_in.end(), h_vec.begin()); // transfer data back to host

    thrust::host_vector<int> v = values, c = counts;
    printf("%d\n", v[v.size() - 1]);
    printf("%d\n", c[c.size() - 1]);
    printf("%.3lf\n", ms);

    // // vector memory automatically released w/ free() or cudaFree()

    return 0;
}