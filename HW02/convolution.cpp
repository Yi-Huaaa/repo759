#include <stdio.h>
#include <stdlib.h>
#include "convolution.h"

#define cov_idx(ori_idx,i,b1) (ori_idx+i+b1)

float val (const float *image, const int x, const int y, const int n, const int m, const int b1, const int b2) {
    // in boundary, return f(x, y)
    if ((x >= 0) && (x < n) && (y >= 0) && (y < n)){
        return image[x*n+y];
    } else if (((x == b1) && (y == b1)) || ((x == b1) && (y == b2)) || ((x == b2) && (y == b1)) || ((x == b2) && (y == b2))){
        return 0;
    } else {
        return 1;
    }
}

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m) {
    int b1 = -((m-1)>>1), b2 = n-b1-1;
    
    //inner
    for (int x = 0; x < n; x++)
        for (int y = 0; y < n; y++)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < m; j++)
                    output[x*n+y] += mask[i*m+j]*val(image, cov_idx(x,i,b1), cov_idx(y,j,b1), n, m, b1, b2); // g (x, y)
}
