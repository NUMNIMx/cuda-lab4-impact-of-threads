#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

#define N 1000000

// CUDA Kernel: RMS Calculation using Grid-Stride Loop
// Using atomicAdd for simplicity in global reduction
__global__ void rmsKernel(float *d_a, float *d_sum_squares, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  float local_sum = 0.0f;

  // Grid-Stride Loop: Allows processing N elements with any number of blocks
  for (int i = idx; i < n; i += stride) {
    float val = d_a[i];
    local_sum += val * val; // Square the element
  }

  // Atomic Add to accumulate results from all threads to a single global
  // variable Note: For very high performance, Block Reduction + Atomic is
  // better, but straight atomicAdd is sufficient for this logic demo.
  atomicAdd(d_sum_squares, local_sum);
}

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main() {
  size_t size = N * sizeof(float);

  // Host Memory
  float *h_a = (float *)malloc(size);

  // Initialize
  srand(time(NULL));
  float cpu_sum_squares = 0.0f;
  for (int i = 0; i < N; i++) {
    h_a[i] = static_cast<float>(rand()) / RAND_MAX;
    cpu_sum_squares += h_a[i] * h_a[i];
  }
  float cpu_rms = sqrt(cpu_sum_squares / N);

  // Device Memory
  float *d_a, *d_sum_squares;
  checkCudaError(cudaMalloc((void **)&d_a, size), "Alloc d_a");
  checkCudaError(cudaMalloc((void **)&d_sum_squares, sizeof(float)),
                 "Alloc d_sum_squares");

  // Copy Data
  checkCudaError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice),
                 "Copy to Device");

  int threadsPerBlock = 256;
  int blockConfigs[] = {32, 64, 128};
  int numConfigs = sizeof(blockConfigs) / sizeof(blockConfigs[0]);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::cout << "Lab 4 Exercise 1.2: RMS Calculation (N = " << N << ")"
            << std::endl;
  std::cout << "--------------------------------------------------------"
            << std::endl;

  // Warmup
  float zero = 0.0f;
  cudaMemcpy(d_sum_squares, &zero, sizeof(float), cudaMemcpyHostToDevice);
  rmsKernel<<<32, 256>>>(d_a, d_sum_squares, N);
  cudaDeviceSynchronize();

  for (int i = 0; i < numConfigs; i++) {
    int blocksPerGrid = blockConfigs[i];

    // Reset sum on device
    cudaMemcpy(d_sum_squares, &zero, sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start);

    // Launch Kernel
    rmsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_sum_squares, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkCudaError(cudaGetLastError(), "Kernel Launch");

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Retrieve result for this run
    float gpu_sum_squares;
    cudaMemcpy(&gpu_sum_squares, d_sum_squares, sizeof(float),
               cudaMemcpyDeviceToHost);
    float gpu_rms = sqrt(gpu_sum_squares / N);

    std::cout << "Blocks: " << blocksPerGrid
              << " | Threads: " << threadsPerBlock
              << " | Time: " << milliseconds << " ms"
              << " | RMS: " << gpu_rms << std::endl;
  }

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "CPU RMS Reference: " << cpu_rms << std::endl;

  // Cleanup
  cudaFree(d_a);
  cudaFree(d_sum_squares);
  free(h_a);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
