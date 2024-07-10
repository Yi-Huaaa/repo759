#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <chrono> // for time
#include <vector>
#include <iostream>
#include <cstring>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

#define start_t(start) start=high_resolution_clock::now(); // Get the starting timestamp
#define end_t(end) end=high_resolution_clock::now(); // Get the ending timestamp
#define duration_t(d,s,e) d=std::chrono::duration_cast<duration<double, std::milli>>(e-s);

void init(float *message_send, float *message_recv, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		message_send[i] = ((float) rand() / RAND_MAX) * 2 - 1; // range [-1, 1];
		message_recv[i] = ((float) rand() / RAND_MAX) * 2 - 1; // range [-1, 1];
	}
}

void print_arr(float *arr, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		printf("%.6lf, ", arr[i]);
	}
	printf("]\n");
}

int main(int argc, char **argv) {
    size_t n = atoi(argv[1]);
 	float *message_send = (float*) malloc(n * sizeof(float));
 	float *message_recv = (float*) malloc(n * sizeof(float));

	// time calculation
    high_resolution_clock::time_point start, end;
    duration<double, std::milli> d2; // duration

    // init
    init(message_send, message_recv, n);

    // MPI settings
    int rank, p, tag = 0;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (rank == 0) {
        int who = 1;

        // start timing t0
	    start_t(start)
        MPI_Send(message_send, n, MPI_FLOAT, who, tag, MPI_COMM_WORLD);
        MPI_Recv(message_recv, n, MPI_FLOAT, who, tag, MPI_COMM_WORLD, &status);
        end_t(end)	
        duration_t(d2, start, end)
        // end timing t0

        // receive time from rank 1
        MPI_Recv(message_recv, 1, MPI_FLOAT, who, tag, MPI_COMM_WORLD, &status);
        
        // print total time
        printf("%lf\n", d2.count() + message_recv[0]);

    } else if (rank == 1) {
        int who = 0;
        
        // start timing t1
	    start_t(start)
        MPI_Recv(message_recv, n, MPI_FLOAT, who, tag, MPI_COMM_WORLD, &status);
        MPI_Send(message_send, n, MPI_FLOAT, who, tag, MPI_COMM_WORLD);
        end_t(end)	
        duration_t(d2, start, end)
        // end timing t1

        message_send[0] = d2.count();

        // send time to rank 0
        MPI_Send(message_send, 1, MPI_FLOAT, who, tag, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}