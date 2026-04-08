g++ -O3 -march=native toric.cpp -o toric_cpu

nvcc toric.cu -o toric_gpu
