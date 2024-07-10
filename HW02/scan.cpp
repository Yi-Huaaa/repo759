#include "scan.h"
#include <stdio.h>
#include <stdlib.h>

void scan(const float *arr, float *output, std::size_t n){
    output[0] = arr [0];
    for (int i = 1; i < n; i++)
        output[i] = output [i-1] + arr[i];       
}