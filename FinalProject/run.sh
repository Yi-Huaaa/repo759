#!/bin/bash

NVCC=/usr/local/cuda-11.8/bin/nvcc

echo "Run task.cu"
# echo "Inputs: x_dir, y_dir, z_dir, n_pin, ITERATIONS"

$NVCC task.cu gamer.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task_gamer
# echo small testcase to check the correctness
# ./task_gamer 8 8 1 4 11
# ./task_gamer 16 16 1 4 11
# echo 64 64 1 4 11
# ./task_gamer 64 64 1 4 11

# # echo 128 128 1 4 11
# ./task_gamer 128 128 1 4 11
# ./task_gamer 256 256 1 4 11
# ./task_gamer 512 512 1 4 11
# ./task_gamer 1024 1024 1 4 11

for n in {4..10} # 16 - 1024
do
    N=$((2 ** n))
    for p in {2..4} # p = {4, 8, 16}
    do  
        P=$((2 ** p))
        echo x_dir = $N, y_dir = $N, z_dir = 1, n_pin = $P, ITERATIONS=11
        ./task_gamer $N $N 1 $P 11
    done 
    echo "-----"
done

rm task_gamer

