#ifndef GAMER_CUH
#define GAMER_CUH


#define IDX23D(x_dir,y_dir,z_dir,x,y,z) (z*x_dir*y_dir+y*x_dir+x) // index to 3D
#define MAX_INF 1073741824 // int MAX: 2147483647, pow(2,30)=1073741824
#define ERR_INF 100000
#define MIN_INF (MAX_INF - ERR_INF)
#define init_bound 0.9

/* Pins structure define */
typedef struct PINS{
    int x = 0;
    int y = 0;
} pins;


/* Timing define*/
#define time_start(start,stop,ms)cudaEventCreate(&start);\
                    cudaEventCreate(&stop);\
                    cudaEventRecord(start);\
                    ms=0.0;

#define time_end(start,stop,ms)cudaEventRecord(stop);\
                    cudaEventSynchronize(stop);\
                    cudaEventElapsedTime(&ms, start, stop);

// Sub-kernels -> construct for testing 
__global__ void prefix_sum (int *output, int *input, int n);
__global__ void prefix_min (int *output, int *input, int n);
__device__ void t_add_s (int *output, int *in1, int *in2);
__device__ void t_add_s (int *output, int *in1, int *in2);


// compute_weight uses device function `prefix_sum_device`
__global__ void compute_weight (int *output_pos, int *output_neg, int *input, int dir_n, int x_dir, int y_dir, int z_dir, int BATCH);
// 
__global__ void find_shortest_pin_set_source (int *gcell_3D_devc, pins *pin_devc, int *set_pin_record, int x_dir, int y_dir, int z_dir, int n_pin, int p, int *path_trace_devc);
// 
__global__ void gamer_x (int *gcell_3D_devc, int *weight_x_post, int *weight_x_nega, int x_dir, int y_dir, int z_dir, int *s_devc, int *t_devc, int *path_trace_devc, int BATCH);
__global__ void gamer_y (int *gcell_3D_devc, int *weight_y_post, int *weight_y_nega, int x_dir, int y_dir, int z_dir, int *s_devc, int *t_devc, int *path_trace_devc, int BATCH);
__global__ void gamer_z (int *gcell_3D_devc, int *weight_x_post, int *weight_x_nega, int x_dir, int y_dir, int z_dir, int *s_devc, int *t_devc, int *path_trace_devc, int BATCH);


#endif
