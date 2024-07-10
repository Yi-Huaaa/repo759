#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "gamer.cuh"
//#define check
// #define TEST


/* Error proofing */
void usage ();
void usage_shared_memory_check (unsigned int s_mem_sz);
int get_GPU_Prop();

/* CPU Functions */
void init (int *gcell_3D_host, int *weight_x_host, int *weight_y_host, int *weight_z_host, pins *pin, int x_dir, int y_dir, int z_dir, int n_pin);
void gen_init_plot (int *gcell_3D_host, int *weight_x_host, int *weight_y_host, int *weight_z_host, pins *pin, int x_dir, int y_dir, int z_dir, int n_pin);

/* GAMER functions */
void gamer_kernel(int *gcell_3D_devc, int *weight_x_devc, int *weight_y_devc, int *weight_z_devc, 
    pins *pin, pins *pin_devc, int *set_pin_record, int x_dir, int y_dir, int z_dir, int n_pin, 
    int ITERATIONS, int *s_devc, int *t_devc, int **weight_ptr, int *path_trace_devc);

/* Test functions */
void test_prefix_sum ();
void test_prefix_min ();
void test_add_min ();

int main(int argc, char* argv[]) {
    if (argc != 6) 
        usage();	
    
    /* Inputs */
    int x_dir = atoi(argv[1]); // x_direction
    int y_dir = atoi(argv[2]); // y_direction
    int z_dir = atoi(argv[3]); // z_direction: if == 1 (2D maze routing)
    int n_pin = atoi(argv[4]); // number of pins: 4, 8, 16
    int ITERATIONS = atoi(argv[5]); // number of iterations: 11

    /* Settings */
    int T_gcell = x_dir*y_dir*z_dir; // size of GCell array 

    /* Ask for memory (Inputs) */
    int *gcell_3D_host = (int*)malloc(T_gcell*sizeof(int)); 
    int *gcell_3D_devc; cudaMalloc((void**)&gcell_3D_devc, T_gcell*sizeof(int));
    // Readin
    int *weight_x_host = (int*)malloc(T_gcell*sizeof(int)); 
    int *weight_x_devc; cudaMalloc((void**)&weight_x_devc, T_gcell*sizeof(int));
    int *weight_y_host = (int*)malloc(T_gcell*sizeof(int)); 
    int *weight_y_devc; cudaMalloc((void**)&weight_y_devc, T_gcell*sizeof(int));
    int *weight_z_host = (int*)malloc(T_gcell*sizeof(int)); 
    int *weight_z_devc; cudaMalloc((void**)&weight_z_devc, T_gcell*sizeof(int));
    pins *pin = (pins*)malloc(n_pin*sizeof(pins)); 
    pins *pin_devc; cudaMalloc((void**)&pin_devc, n_pin*sizeof(pins));
    

    /* Ask for GAMER memory */
    // Weight arrays -> 6
    int **weight_ptr = new int*[6];
    int *weight_x_post; cudaMalloc((void**)&weight_x_post, T_gcell*sizeof(int));
    int *weight_x_nega; cudaMalloc((void**)&weight_x_nega, T_gcell*sizeof(int));
    int *weight_y_post; cudaMalloc((void**)&weight_y_post, T_gcell*sizeof(int));
    int *weight_y_nega; cudaMalloc((void**)&weight_y_nega, T_gcell*sizeof(int));
    int *weight_z_post; cudaMalloc((void**)&weight_z_post, T_gcell*sizeof(int));
    int *weight_z_nega; cudaMalloc((void**)&weight_z_nega, T_gcell*sizeof(int));
    weight_ptr[0] = weight_x_post;
    weight_ptr[1] = weight_x_nega;
    weight_ptr[2] = weight_y_post;
    weight_ptr[3] = weight_y_nega;
    weight_ptr[4] = weight_z_post;
    weight_ptr[5] = weight_z_nega;
    int *s_devc; cudaMalloc((void**)&s_devc, T_gcell*sizeof(int));
    int *t_devc; cudaMalloc((void**)&t_devc, T_gcell*sizeof(int)); // temp array
    int *set_pin_0 = (int*)malloc(n_pin*sizeof(int)); memset(set_pin_0, 0, n_pin*sizeof(int));
    int *set_pin_record; cudaMalloc((void**)&set_pin_record, n_pin*sizeof(int));
    int *path_trace_host = (int*)malloc(T_gcell*sizeof(int)); //memset(path_trace_host, 0, T_gcell*sizeof(int));
    for (auto ii = 0; ii < T_gcell; ii ++) {
        path_trace_host[ii] = -1;
    }
    int *path_trace_devc; cudaMalloc((void**)&path_trace_devc, T_gcell*sizeof(int));


    /* Timing */
    cudaEvent_t start;
    cudaEvent_t stop;
    float ms = 0.0;

    /* Initialization */
    srand(time(NULL));
    init(gcell_3D_host, weight_x_host, weight_y_host, weight_z_host, pin, x_dir, y_dir, z_dir, n_pin);    
    // gen_init_plot(gcell_3D_host, weight_x_host, weight_y_host, weight_z_host, pin, x_dir, y_dir, z_dir, n_pin);

    /* Data movement */
    cudaMemcpy(gcell_3D_devc, gcell_3D_host, T_gcell*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_x_devc, weight_x_host, T_gcell*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_y_devc, weight_y_host, T_gcell*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_z_devc, weight_z_host, T_gcell*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pin_devc, pin, n_pin*sizeof(pins), cudaMemcpyHostToDevice);
    cudaMemcpy(set_pin_record, set_pin_0, n_pin*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(path_trace_devc, path_trace_host, T_gcell*sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();


    /* Test function */
#ifdef TEST
    /* Testing funcions */
    time_start(start, stop, ms)    
        // test_prefix_sum();
        // test_prefix_min();
        // test_add_min();
    time_end(start,stop, ms)
    // printf("ms = %.3lf\n", ms);

#else    
    /* Start GAMER */
    time_start(start, stop, ms)
        gamer_kernel(gcell_3D_devc, weight_x_devc, weight_y_devc, weight_z_devc, pin, pin_devc, set_pin_record, x_dir, y_dir, z_dir, n_pin, ITERATIONS, s_devc, t_devc, weight_ptr, path_trace_devc);
    time_end(start,stop, ms)

    printf("time = %.3lf (ms)\n", ms);
    cudaMemcpy(gcell_3D_host, gcell_3D_devc, T_gcell*sizeof(int), cudaMemcpyDeviceToHost);

#endif

    free (gcell_3D_host);
    free (weight_x_host);
    free (weight_y_host);
    free (weight_z_host);
    free (set_pin_0);
    free (path_trace_host); 
    cudaFree (gcell_3D_devc);
    cudaFree (weight_x_devc);
    cudaFree (weight_y_devc);
    cudaFree (weight_z_devc);
    cudaFree (s_devc);
    cudaFree (t_devc);
    cudaFree (pin_devc);
    cudaFree (set_pin_record);
    cudaFree (path_trace_devc);
    for (auto i = 0; i < 6; i++) 
        cudaFree (weight_ptr[i]);


    return 0;
}


void usage () {
    printf("Usage:\n");
    printf("       ./task x_dir y_dir z_dir n_pin ITERATIONS\n");
    exit(0);
}

void usage_shared_memory_check (unsigned int s_mem_sz) {
    unsigned int s_mem_bound = 48*1024;
    if (s_mem_sz > s_mem_bound) {
        printf("Shared memory size out of bound, now: %u (bound: 48 (KB))\n", s_mem_sz);
        exit(0);
    }
}

// todo: init function -> simplify
void init (int *gcell_3D_host, int *weight_x_host, int *weight_y_host, int *weight_z_host, pins *pin, int x_dir, int y_dir, int z_dir, int n_pin) {
    // number of pins 
#ifdef check
    pin[0].x = 0;
    pin[0].y = 0;
    for (auto p = 1; p < (n_pin); p++) {
        pin[p].x = (n_pin-p);
        pin[p].y = 0;
    }
#else 
    // gen random pins
    for (auto p = 0; p < (n_pin); p++) {
        pin[p].x = rand()%(x_dir);
        pin[p].y = rand()%(y_dir);
    }
    // todo: check whetehr duplicated points   
#endif
    // GCell array     
    for (auto z = 0; z < z_dir; z++) {
        for (auto x = 0; x < x_dir; x++) {
            for (auto y = 0; y < y_dir; y++) {
                gcell_3D_host[IDX23D(x_dir,y_dir,z_dir,x,y,z)] = MAX_INF;
            }
        }
    }

    // weight array: x dimension
#ifndef check
    float wt = 0.0, wt_c = 0.0; // random number 
#endif
    for (auto z = 0; z < (z_dir); z++) {
        for (auto y = 0; y < (y_dir); y++) {
            for (auto x = 0; x < (x_dir); x++) {
#ifdef check
                weight_x_host[IDX23D((x_dir),(y_dir),(z_dir),x,y,z)] = 1;
#else 
                wt = (int)ceil((((float)rand()/RAND_MAX))*3); // range [0, 3] -> ceiling to {1, 2, 3}
                wt_c = ((float)rand()/RAND_MAX); // [0, 1]
                if (wt_c > init_bound) {
                    wt_c = (int)ceil((((float)rand()/RAND_MAX))*9); // range [0, 9] -> ceiling to {1, ..., 9}
                    wt = 3+(pow(2, wt_c));
                }
                weight_x_host[IDX23D((x_dir),(y_dir),(z_dir),x,y,z)] = wt;
#endif
                if (x == (x_dir-1)) {
                    weight_x_host[IDX23D((x_dir),(y_dir),(z_dir),x,y,z)] = 0;
                }
            }
        }
    }
    // weight array: y dimension
    // Note: readin transpose     
    for (auto z = 0; z < (z_dir); z++) {
        for (auto y = 0; y < (y_dir); y++) {
            for (auto x = 0; x < (x_dir); x++) {
#ifdef check
                weight_y_host[IDX23D((x_dir),(y_dir),(z_dir),x,y,z)] = 1;
#else 
                wt = (int)ceil((((float)rand()/RAND_MAX))*3); // range [0, 3] -> ceiling to {1, 2, 3}                
                wt_c = ((float)rand()/RAND_MAX); // [0, 1]
                if (wt_c > init_bound) {
                    wt_c = (int)ceil((((float)rand()/RAND_MAX))*9); // range [0, 9] -> ceiling to {1, ..., 9}
                    wt = 3+(pow(2, wt_c));
                }
                weight_y_host[IDX23D((x_dir),(y_dir),(z_dir),x,y,z)] = wt;
#endif
                if (x == (x_dir-1)) {
                    weight_y_host[IDX23D((x_dir),(y_dir),(z_dir),x,y,z)] = 0;
                }
            }
        }
    }
    // weight array: z dimension
    // Note: readin transpose   
    for (auto z = 0; z < (z_dir); z++) {
        for (auto y = 0; y < (y_dir); y++) {
            for (auto x = 0; x < (x_dir); x++) {
#ifdef check
                weight_z_host[IDX23D((x_dir),(y_dir),(z_dir),x,y,z)] = 1;
#else 
                wt = (int)ceil((((float)rand()/RAND_MAX))*3); // range [0, 3] -> ceiling to {1, 2, 3}                
                wt_c = ((float)rand()/RAND_MAX); // [0, 1]
                if (wt_c > init_bound) {
                    wt_c = (int)ceil((((float)rand()/RAND_MAX))*9); // range [0, 9] -> ceiling to {1, ..., 9}
                    wt = 3+(pow(2, wt_c));
                }
                weight_z_host[IDX23D((x_dir),(y_dir),(z_dir),x,y,z)] = wt;
#endif
                if (z == (z_dir-1)) {
                    weight_z_host[IDX23D((x_dir),(y_dir),(z_dir),x,y,z)] = 0;
                }
            }
        }
    }
}

void gen_init_plot (int *gcell_3D_host, int *weight_x_host, int *weight_y_host, int *weight_z_host, pins *pin, int x_dir, int y_dir, int z_dir, int n_pin) {
    // pins
    printf("n_pin = %d\n", n_pin);
    for (auto p = 0; p < n_pin; p++) {
        printf("pins[%d]: x = %d, y = %d\n", p, pin[p].x, pin[p].y);
    }

    // GCell array
    printf("GCell array, with x_dir = %d, y_dir = %d, z_dir = %d\n", x_dir, y_dir, z_dir);
    for (auto z = 0; z < z_dir; z++) {
        printf("Metal layer %d\n", z);
        for (auto y = 0; y < y_dir; y++) {
            for (auto x = 0; x < x_dir; x++) {
                if (gcell_3D_host[IDX23D(x_dir,y_dir,z_dir,x,y,z)] == MAX_INF) {
                    printf("INF, ");
                } else {
                    printf("GCell initial ERROR\n");
                    exit(0);
                }
            }
            printf("\n");
        }
        printf("\n");
    }

    // todo: merge all of the print array into a function
    // weight array
    printf("Weight array: x dimension\n");
    for (auto z = 0; z < (z_dir); z++) {
        printf("Metal layer %d -> %d\n", z, (z+1));
        for (auto y = 0; y < (y_dir); y++) {
            for (auto x = 0; x < (x_dir); x++) {
                printf("%d, ", weight_x_host[IDX23D((x_dir),(y_dir),(z_dir),x,y,z)]);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("Weight array: y dimension\n");
    for (auto z = 0; z < (z_dir); z++) {
        printf("Metal layer %d -> %d\n", z, (z+1));
        for (auto y = 0; y < (y_dir); y++) {
            for (auto x = 0; x < (x_dir); x++) {
                printf("%d, ", weight_y_host[IDX23D((x_dir),(y_dir),(z_dir),x,y,z)]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // printf("Weight array: z dimension\n");
    // for (auto z = 0; z < (z_dir); z++) {
    //     printf("Metal layer %d -> %d\n", z, (z+1));
    //     for (auto y = 0; y < (y_dir); y++) {
    //         for (auto x = 0; x < (x_dir); x++) {
    //             printf("%d, ", weight_z_host[IDX23D((x_dir),(y_dir),(z_dir),x,y,z)]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
}

void test_prefix_sum () {
    printf("Test prefix_sum\n");
    int len = 512;
    int *input = (int*)malloc(len*sizeof(int)); 
    int *output = (int*)malloc(len*sizeof(int)); 
    for (auto i = 0; i < len; i++) {
        input[i] = 1;
        output[i] = 0;
    }

    int *input_gpu;  cudaMalloc((void**)&input_gpu, len*sizeof(int));
    int *output_gpu; cudaMalloc((void**)&output_gpu, len*sizeof(int));
    
    cudaMemcpy(input_gpu, input, len*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(output_gpu, output, len*sizeof(int), cudaMemcpyHostToDevice);
    
    
    unsigned int smem_sz = len*sizeof(int);
    prefix_sum <<< 1, (len/2), smem_sz >>> (output_gpu, input_gpu, len);
    cudaDeviceSynchronize();

    cudaMemcpy(output, output_gpu, len*sizeof(int), cudaMemcpyDeviceToHost);


    // printf("prefix_sum resutls = [");
    // for (auto i = 0; i < len; i++) {
    //     printf("%d, ", output[i]);
    // } printf("]\n");
}

void test_prefix_min () {
    printf("Test prefix_min\n");
    int len = 16;
    int *input = (int*)malloc(len*sizeof(int)); 
    int *output = (int*)malloc(len*sizeof(int)); 
    for (auto i = 0; i < len/2; i++) {
        input[i] = len-i + 2;
        output[i] = 0;
    }
    for (auto i = len/2; i < len; i++) {
        input[i] = i;
        output[i] = 0;
    }

    int *input_gpu;  cudaMalloc((void**)&input_gpu, len*sizeof(int));
    int *output_gpu; cudaMalloc((void**)&output_gpu, len*sizeof(int));
    
    cudaMemcpy(input_gpu, input, len*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(output_gpu, output, len*sizeof(int), cudaMemcpyHostToDevice);
    
    unsigned int smem_sz = len*sizeof(int);
    prefix_min <<< 1, (len/2), smem_sz >>> (output_gpu, input_gpu, len);
    cudaDeviceSynchronize();

    cudaMemcpy(output, output_gpu, len*sizeof(int), cudaMemcpyDeviceToHost);


    printf("prefix_min resutls = [");
    for (auto i = 0; i < len; i++) {
        printf("%d, ", output[i]);
    } printf("]\n");
}

void gamer_kernel (int *gcell_3D_devc, int *weight_x_devc, int *weight_y_devc, int *weight_z_devc, 
    pins *pin, pins *pin_devc, int *set_pin_record, int x_dir, int y_dir, int z_dir, int n_pin, 
    int ITERATIONS, int *s_devc, int *t_devc, int **weight_ptr, int *path_trace_devc) {
    
    int z = 1;
    // Note: this BATCH only used for "2D maze routing" (x_dir == y_dir), at most used 48 KB shared memory
    // one time deal with 8192 elements
    int BATCH = (x_dir > 64) ? ((x_dir*y_dir*z)/8192) : (1); // /64

    // compute s(i) = sigma(c(i))
        // Note: Due to no enough shared memory, we need to deal the total computation by batch
    // printf("BATCH = %d (%d), s_mem = %d, #blk = %d, #thd = %d\n", BATCH, x_dir*y_dir*z_dir, x_dir*(y_dir/BATCH)*z_dir, ((y_dir/BATCH)*z_dir), (x_dir/2));
    // x+, x-
    compute_weight <<< ((y_dir/BATCH)*z_dir), (x_dir/2), x_dir*(y_dir/BATCH)*z_dir*sizeof(int) >>> (weight_ptr[0], weight_ptr[1], weight_x_devc, x_dir, x_dir, y_dir, z_dir, BATCH);
    // y+, y-
    compute_weight <<< ((x_dir/BATCH)*z_dir), (y_dir/2), (x_dir/BATCH)*y_dir*z_dir*sizeof(int) >>> (weight_ptr[2], weight_ptr[3], weight_y_devc, y_dir, x_dir, y_dir, z_dir, BATCH);    
#ifdef check
    printf("Ask shared memory = %lu (KB), with BATCH = %d\n",  (x_dir*(y_dir/BATCH)*z_dir*sizeof(int))/1024, BATCH);
#endif

/*
    // test compute_weight, python test PASS
    cudaDeviceSynchronize();
    int T_gcell = x_dir*y_dir*z_dir;
    int *tmp1 = (int*)malloc(T_gcell*sizeof(int)); 
    cudaMemcpy(tmp1, weight_ptr[0], T_gcell*sizeof(int), cudaMemcpyDeviceToHost);
    printf("After prefix_sum, x positive\n");
    for (auto z = 0; z < (z_dir); z++) {
        printf("Metal layer %d -> %d\n", z, (z+1));
        for (auto y = 0; y < (y_dir); y++) {
            for (auto x = 0; x < (x_dir); x++) {
                printf("%d, ", tmp1[IDX23D((x_dir),(y_dir),(z_dir),x,y,z)]);
            }
            printf("\n");
        }
        printf("\n");
    }
    int *tmp2 = (int*)malloc(T_gcell*sizeof(int)); 
    cudaMemcpy(tmp2, weight_ptr[1], T_gcell*sizeof(int), cudaMemcpyDeviceToHost);
    printf("After prefix_sum, x negative\n");
    for (auto z = 0; z < (z_dir); z++) {
        printf("Metal layer %d -> %d\n", z, (z+1));
        for (auto y = 0; y < (y_dir); y++) {
            for (auto x = 0; x < (x_dir); x++) {
                printf("%d, ", tmp2[IDX23D((x_dir),(y_dir),(z_dir),x,y,z)]);
            }
            printf("\n");
        }
        printf("\n");
    }
    int *tmp3 = (int*)malloc(T_gcell*sizeof(int)); 
    cudaMemcpy(tmp3, weight_ptr[2], T_gcell*sizeof(int), cudaMemcpyDeviceToHost);
    printf("After prefix_sum, y positive\n");
    for (auto z = 0; z < (z_dir); z++) {
        printf("Metal layer %d -> %d\n", z, (z+1));
        for (auto y = 0; y < (y_dir); y++) {
            for (auto x = 0; x < (x_dir); x++) {
                printf("%d, ", tmp3[IDX23D((x_dir),(y_dir),(z_dir),x,y,z)]);
            }
            printf("\n");
        }
        printf("\n");
    }
    int *tmp4 = (int*)malloc(T_gcell*sizeof(int)); 
    cudaMemcpy(tmp4, weight_ptr[3], T_gcell*sizeof(int), cudaMemcpyDeviceToHost);
    printf("After prefix_sum, y negative\n");
    for (auto z = 0; z < (z_dir); z++) {
        printf("Metal layer %d -> %d\n", z, (z+1));
        for (auto y = 0; y < (y_dir); y++) {
            for (auto x = 0; x < (x_dir); x++) {
                printf("%d, ", tmp4[IDX23D((x_dir),(y_dir),(z_dir),x,y,z)]);
            }
            printf("\n");
        }
        printf("\n");
    }
*/
    // GAMER iterations
    // int z = 0;
    for (auto p = 0; p < n_pin; p++) {
        // find the pin with the shortest path -> connect that pin and set source distance to 0
        // todo: merge this kernel into other kernels -> to reduce the kernel launching time
        // todo: the setting of thread = x_dir may be wrong when x_dir > 1024 -> will launch kernel fail
        find_shortest_pin_set_source <<< 1, 1 >>> (gcell_3D_devc, pin_devc, set_pin_record, x_dir, y_dir, z_dir, n_pin, p, path_trace_devc);
        
        for (auto itr = 0; itr < ITERATIONS; itr ++) {
            // Note: Due to no enough shared memory, we need to deal the total computation by batch
            // x direction
            gamer_x <<< ((y_dir/BATCH)*z_dir), (x_dir/2), x_dir*(y_dir/BATCH)*z_dir*sizeof(int) >>> (gcell_3D_devc, weight_ptr[0], weight_ptr[1], x_dir, y_dir, z_dir, s_devc, t_devc, path_trace_devc, BATCH);
            // y direction
            gamer_y <<< ((x_dir/BATCH)*z_dir), (y_dir/2), (x_dir/BATCH)*y_dir*z_dir*sizeof(int) >>> (gcell_3D_devc, weight_ptr[2], weight_ptr[3], x_dir, y_dir, z_dir, s_devc, t_devc, path_trace_devc, BATCH);
            // // z direction
            // gamer_z <<< 1, 1 >>> ();
        }
/*
                // test gamer_x and gamer_y
                cudaDeviceSynchronize();
                int *check_matrix = gcell_3D_devc; // ! rewrite this index to printf
                
                int T_gcell = x_dir*y_dir*z_dir;
                int *tmp1 = (int*)malloc(T_gcell*sizeof(int)); 
                cudaMemcpy(tmp1, check_matrix, T_gcell*sizeof(int), cudaMemcpyDeviceToHost);
                // printf("itr = %d, After prefix_min, gamer_y (connected pin %d)\n", itr, p);
                for (auto z = 0; z < (z_dir); z++) {
                    printf("Metal layer %d -> %d\n", z, (z+1));
                    for (auto y = 0; y < (y_dir); y++) {
                        for (auto x = 0; x < (x_dir); x++) {
                            int val = tmp1[IDX23D((x_dir),(y_dir),(z_dir),x,y,z)];
                            if (val >= MIN_INF) {
                                printf("INF, ");
                            } else {
                                printf("%d, ", val);
                            }
                        }
                        printf("\n");
                    }
                    printf("\n");
                }

                check_matrix = path_trace_devc;
                cudaMemcpy(tmp1, check_matrix, T_gcell*sizeof(int), cudaMemcpyDeviceToHost);
                // printf("itr = %d, After prefix_min, gamer_y (connected pin %d)\n", itr, p);
                for (auto z = 0; z < (z_dir); z++) {
                    printf("(path_trace_devc): Metal layer %d -> %d\n", z, (z+1));
                    for (auto y = 0; y < (y_dir); y++) {
                        for (auto x = 0; x < (x_dir); x++) {
                            int val = tmp1[IDX23D((x_dir),(y_dir),(z_dir),x,y,z)];
                            if (val >= MIN_INF) {
                                printf("INF, ");
                            } else {
                                printf("%d, ", val);
                            }
                        }
                        printf("\n");
                    }
                    printf("\n");
                }
                printf("--------\n\n");
*/ 
    }

}



int get_GPU_Prop() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("deviceProp.clockRate = %d\n", deviceProp.clockRate);
    printf("deviceProp.totalGlobalMem = %zu\n", deviceProp.totalGlobalMem);
    printf("deviceProp.warpSize = %d\n", deviceProp.warpSize);
    printf("deviceProp.totalConstMem = %zu\n", deviceProp.totalConstMem);
    printf("deviceProp.canMapHostMemory = %d\n", deviceProp.canMapHostMemory);
    printf("deviceProp.minor = %d\n", deviceProp.minor); // Minor compute capability, e.g. cuda 9.0

    // about shared memory 
    printf("\n");
    printf("deviceProp.sharedMemPerBlockOptin = %zu\n", deviceProp.sharedMemPerBlockOptin);
    printf("deviceProp.sharedMemPerBlock = %zu\n", deviceProp.sharedMemPerBlock);
    printf("deviceProp.sharedMemPerMultiprocessor = %zu\n", deviceProp.sharedMemPerMultiprocessor);
    
    // about SM and block
    printf("\n");
    printf("deviceProp.multiProcessorCount = %d\n", deviceProp.multiProcessorCount);
    printf("deviceProp.maxBlocksPerMultiProcessor = %d\n", deviceProp.maxBlocksPerMultiProcessor);
    printf("deviceProp.maxThreadsPerBlock = %d\n", deviceProp.maxThreadsPerBlock);
    printf("deviceProp.maxThreadsPerMultiProcessor = %d\n", deviceProp.maxThreadsPerMultiProcessor);
    
    // about registers  
    printf("\n");  
    printf("deviceProp.regsPerBlock = %d\n", deviceProp.regsPerBlock);
    printf("deviceProp.regsPerMultiprocessor = %d\n", deviceProp.regsPerMultiprocessor);

    // something
    printf("\n");
    printf("deviceProp.maxGridSize[0] = %d, deviceProp.maxGridSize[1] = %d, deviceProp.maxGridSize[2] = %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    return deviceProp.clockRate;
}
