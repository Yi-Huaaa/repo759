#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>

#include <omp.h>


int factorial_of_integers (int input, int myId) {
    int ret = 1;
    for (auto i = 2; i < (input+1); i++) {
        ret *= i;
    }
    // printf("input = %d, by which thread: %d\n", input, myId);
    return ret;
}

int main(int argc, char* argv[]){
    omp_set_num_threads(4); 

    int nThreads, tid = 4;
    #pragma omp parallel 
    #pragma omp master 
    {
        nThreads = omp_get_num_threads();
        printf("Number of threads: %d\n", nThreads);
    }


    #pragma omp parallel private (tid)
    {
        tid = omp_get_thread_num();
        
        printf("I am thread No. %d\n", tid);

        for (int i = tid+1; i < 2*nThreads+1; i+=4) {
            int ans = factorial_of_integers(i, tid);
            printf("%d!=%d\n", i, ans);
            // printf("%d!=%d, by threads = %d\n", i, ans, tid);
        }
    }
}