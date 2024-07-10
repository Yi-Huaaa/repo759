# FinalProject
## Run code
```
bash run.sh
```

## Milestones
### 2D maze routing 

| Date  | Achievements |
| ----- | ------------ |
| 11/20 (Mon)  | Initialization  |
| 11/21 (Tue)  | Add `prefix_sum` `prefix_min` `d_min_s` `t_add_s` kernel PASS  |
| 11/24 (Fri)  | `gamer_x` kernel PASS (debug in `d_min_s`, `prefix_min_device`, `t_add_s`)   |
| 11/25 (Sat)  | update `compute_weight`, only need to compute `prefix_sum_device` for only 1 time, since we can replace the negative direction `prefix_sum_device` by using the MAX_results in positive direction `prefix_sum_device` to minus the inplace number to get same result of negative direction `prefix_sum_device`|
| 11/28 (Tue)  | (1) Successful - 2D GAMER, however, when `n >= 128`, shared memory not enough, hence need to be batch (next version), (2) debug -> remove the time consuming code in both `gamer_x` and `gamer_y` |
| 11/28 (Tue)  | (1) Fix when `n >= 128`, shared memory not enough problem by using `batching` |
| 11/29 (Wed)  | (1) Add `path_trace_host` `path_trace_devc` array to record where the shortest path comes from |
| 11/29 (Wed)  | (1) Debug in `find_shortest_pin_set_source`, successfully print the connected path in a serially connected `0` |
| 12/10 (Sun)  | (1) update `compute_weight`, `gamer_x` and `gamer_y`: Moving the batching into the kernel -> reduce the kernel launching time |



# Note: 
* prefix_sum 
* 11/25 gamer_x: 64*64 launch kernel success, 128x128kernel launch failed;
    * Since sharedMemPerBlock = 48 KB, however, as (x_dir, y_dir, z_dir) = (128, 128, 1), ask for 128 * 128 * 1 * 4 (B) = 64 KB shared memory per block, which is larger than 48 KB, thus failed
* 11/28 ori code in `gamer_x` and `gamer_y`: int **w_array = new int *[2];  $\to$ new code: int *w_array[2];
    * The original code will ask memory from heap (very far memory), also forgot to free after the kernel finish. The two things make the program cost lots of time. The new code just ask a local variable on stack $\to$ faster, also can be optimizaed by compiler, therefore, program becomes faster.
* 11/28 Fix when `n >= 128`, shared memory not enough problem by using `batching`
    * More optimization: move the batching `for loop` into kernel $\to$ reduce the time to launching kernel
* 11/29 Recording the path tracing: `path_trace_devc` array. initialized with zero. When shortest path passes from (right (x+), left (x-), upper (y+), lower (y-), top (z+), down (z-)) = (1, 2, 3, 4, 5, 6).
* 11/29 Debug in `find_shortest_pin_set_source` kernel, when updating the shortest path, all the pins on the shortest path need to be set as `0` (for both `gcell_3D_devc` and `path_trace_devc` array) $\to$ NOTE: now only use 1 thread to trace the shortest path, for more optimization, myabe it will be better to use multi-thread, however, the time seems the same while only use 1 thread to update the shortest path.
* 12/10 update `compute_weight`, `gamer_x` and `gamer_y`: Moving the batching into the kernel -> reduce the kernel launching time


# Todo
1. (V) (11/30) Remove `t_devc` and `s_devc`, making all the computation of gamer run in the shared memory -> Testing time (v2) $\to$ remain the original case would be faster due to the bank comflict in the shared memory 
2. (11/30) Lots of todo for optimization in the program
3. (11/30) Moving the batching into the kernel -> reduce the kernel launching time -> Testing time (v3)
4. (11/30) Warp primitive
5. (12/01) Add z-axils for the 3D maze routing (12/01) Add z-axils for the 3D maze routing 
6. (12/01) Parse in the FrontEnd

---

# Runtime Record 
* Original version (w/o shared memory)

|  Grid Graph Size | #Pins | 1129_Runtime (ms) | 
| ---------------- | ----- | ------------ |
| 16 | 4  |  1.025 |
| 16 | 8  |  2.039 |
| 16 | 16 |  4.033 |
| 32 | 4  |  1.206 |
| 32 | 8  |  2.403 |
| 32 | 16 |  4.830 |
| 64 | 4  |  1.410 |
| 64 | 8  |  2.831 |
| 64 | 16 |  5.650 |
| 128 | 4  |  3.112 |
| 128 | 8  |  6.192 |
| 128 | 16 |  12.367 |
| 256 | 4  |  13.897 |
| 256 | 8  |  27.699 |
| 256 | 16 |  55.296 |
| 512 | 4  |  57.132 |
| 512 | 8  |  113.639 |
| 512 | 16 |  225.828 |
| 1024 | 4  |  282.665 |
| 1024 | 8  |  562.888 |
| 1024 | 16 |  1124.624 |

* 11/28 version - 2D maze routing 

|  Grid Graph Size | #Pins | 1129_Runtime (ms) | 
| ---------------- | ----- | ------------ |
| 16 | 4  |   0.425   |
| 16 | 8  |   0.839   |
| 16 | 16 |   1.679   |
| 32 | 4  |   0.475   |
| 32 | 8  |   0.936   |
| 32 | 16 |   1.882   |
| 64 | 4  |   0.542   |
| 64 | 8  |   1.069   |
| 64 | 16 |   2.119   |
| 128 | 4  |   1.340   |
| 128 | 8  |   2.606   |
| 128 | 16 |   5.181   |
| 256 | 4  |   5.850   |
| 256 | 8  |   11.479   |
| 256 | 16 |   22.607   |
| 512 | 4  |   30.938   |
| 512 | 8  |   60.893   |
| 512 | 16 |   115.355   |
| 1024 | 4  |   143.672   |
| 1024 | 8  |   283.867   |
| 1024 | 16 |   554.491   |

* 12/10 version - 2D maze routing 

|  Grid Graph Size | #Pins | 1210_Runtime (ms) | 
| ---------------- | ----- | ------------ |
| 16 | 4  |  0.444 |
| 16 | 8  |  0.900 |
| 16 | 16 |  1.783 |
| 32 | 4  |  0.502 |
| 32 | 8  |  0.989 |
| 32 | 16 |  1.975 |
| 64 | 4  |  0.563 |
| 64 | 8  |  1.109 |
| 64 | 16 |  2.194 |
| 128 | 4  |  1.166 |
| 128 | 8  |  2.301 |
| 128 | 16 |  4.589 |
| 256 | 4  |  4.770 |
| 256 | 8  |  9.485 |
| 256 | 16 |  18.934 |
| 512 | 4  |  24.935 |
| 512 | 8  |  49.688 |
| 512 | 16 |  98.373 |
| 1024 | 4  | 118.686 |
| 1024 | 8  | 236.090 |
| 1024 | 16 |  471.754 |
