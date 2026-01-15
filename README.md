# Lab 4: The Impact of Threads and Blocks on Processing Speed

This repository contains CUDA C++ implementations for Lab 4, analyzing the performance impact of different thread and block configurations on GPU processing speed.

## Files

1.  **`lab4_1_array_add.cu` (Exercise 1.1: Array Addition)**
    *   Performs vector addition on arrays of size $N=10^6$.
    *   Compares performance across 128, 256, and 512 threads per block.
    *   Includes warmup kernel and connection verification.

2.  **`lab4_2_rms.cu` (Exercise 1.2: Root Mean Square)**
    *   Calculates RMS of an array ($N=10^6$).
    *   Uses a **Grid-Stride Loop** and `atomicAdd` to handle arbitrary block sizes.
    *   Compares performance with fixed threads (256) and varying blocks (32, 64, 128).

3.  **`lab4_3_sum_array.cu` (Exercise 1.3: SumArray CPU vs GPU)**
    *   Sums a large array ($N=10^7$).
    *   Benchmarks GPU performance against CPU execution time.
    *   Tests specific thread/block scenarios requested in the lab sheet (Question 1-3).

## Compilation & Usage

Each file is standalone and can be compiled using `nvcc`.

### 1. Array Addition
```bash
nvcc lab4_1_array_add.cu -o lab4_1_array_add
./lab4_1_array_add
```

### 2. RMS Calculation
```bash
nvcc lab4_2_rms.cu -o lab4_2_rms
./lab4_2_rms
```

### 3. SumArray Benchmark
```bash
nvcc lab4_3_sum_array.cu -o lab4_3_sum_array
./lab4_3_sum_array
```

## Requirements
*   NVIDIA GPU with CUDA capability
*   CUDA Toolkit (nvcc compiler)
