#include <stdio.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "count.cuh"

#include <iostream>
using namespace std;

void host_print_arr (int values[7]) {
    for (size_t i = 0; i < 7; i++) {
        std::cout << values[i] << '\n';
    }
    std::cout << "\n\n";
}

void dev_print_arr (thrust::device_vector<int>& values) {
    for (size_t i = 0; i < values.size(); i++) {
        std::cout << values[i] << ' ';
    }
    std::cout << std::endl;
}

void count(const thrust::device_vector<int>& d_in, thrust::device_vector<int>& values, thrust::device_vector<int>& counts) {
    
    // copy the input to values
    values = d_in;
    // dev_print_arr(values);
    
    // sort the values
    thrust::sort(values.begin(), values.end());
    // dev_print_arr(values);
    
    // resize the vector counts and fill with ones
    counts.resize(values.size(), 1);
    // dev_print_arr(counts);

    // reduce
    auto new_end = thrust::reduce_by_key(values.begin(), values.end(), counts.begin(), values.begin(), counts.begin());
    values.resize(new_end.first - values.begin());
    counts.resize(new_end.second - counts.begin());
    // dev_print_arr(values);
    // dev_print_arr(counts);

    // const int N = 7;
    // int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
    // int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
    // int C[N];                         // output keys
    // int D[N];                         // output values
    // thrust::pair<int*,int*> new_end;
    // new_end = thrust::reduce_by_key(thrust::host, A, A + N, B, C, D);
    // // The first four keys in C are now {1, 3, 2, 1} and new_end.first - C is 4.
    // // The first four values in D are now {9, 21, 9, 3} and new_end.second - D is 4.    

    /*
        input:   1, 3, 4, 2, 2, 2, 3, 5, 5
        sort:    1, 2, 2, 2, 3, 3, 4, 5, 5
        ones:    1, 1, 1, 1, 1, 1, 1, 1, 1
        
        reduce by key: (input: sorting array, ones)
        unique:  1, 2, 3, 4, 5
        count:   1, 3, 
    */
}
