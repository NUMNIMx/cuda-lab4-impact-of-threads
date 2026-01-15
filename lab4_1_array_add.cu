#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

// Define Array Size N = 10^6
#define N 1000000

// CUDA Kernel for Array Addition
__global__ void addKernel(float *d_a, float *d_b, float *d_c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    d_c[idx] = d_a[idx] + d_b[idx];
  }
}

// Helper function to check CUDA errors
void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main() {
  // Size in bytes
  size_t size = N * sizeof(float);

  // Host memory allocation
  float *h_a = (float *)malloc(size);
  float *h_b = (float *)malloc(size);
  float *h_c = (float *)malloc(size);

  // Initialize arrays with random values
  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_a[i] = static_cast<float>(rand()) / RAND_MAX;
    h_b[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Device memory allocation
  float *d_a, *d_b, *d_c;
  checkCudaError(cudaMalloc((void **)&d_a, size), "Allocating d_a");
  checkCudaError(cudaMalloc((void **)&d_b, size), "Allocating d_b");
  checkCudaError(cudaMalloc((void **)&d_c, size), "Allocating d_c");

  // Copy data from Host to Device
  checkCudaError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice),
                 "Copying h_a to d_a");
  checkCudaError(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice),
                 "Copying h_b to d_b");

  // Configurations to test
  int threadConfigs[] = {128, 256, 512};
  int numConfigs = sizeof(threadConfigs) / sizeof(threadConfigs[0]);

  std::cout << "Lab 4 Exercise 1.1: Array Addition (N = " << N << ")"
            << std::endl;
  std::cout << "--------------------------------------------------------"
            << std::endl;

  // Create CUDA Events for timing
  cudaEvent_t start, stop;
  checkCudaError(cudaEventCreate(&start), "Creating start event");
  checkCudaError(cudaEventCreate(&stop), "Creating stop event");

  // Warmup Kernel (to absorb cold start overhead)
  addKernel<<<(N + 128 - 1) / 128, 128>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();

  for (int i = 0; i < numConfigs; i++) {
    int threadsPerBlock = threadConfigs[i];

    // Calculate Blocks needed: (N + Threads - 1) / Threads
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Record start time
    cudaEventRecord(start);

    // Launch Kernel
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // Wait for the GPU to finish

    // Check for kernel errors
    checkCudaError(cudaGetLastError(), "Kernel Launch");

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Threads per Block: " << threadsPerBlock
              << " | Blocks: " << blocksPerGrid << " | Time: " << milliseconds
              << " ms" << std::endl;
  }

  // Verify Results (Check the last run's output)
  checkCudaError(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost),
                 "Copying d_c to h_c");
  bool correct = true;
  for (int i = 0; i < N; i++) {
    // Use a small epsilon for float comparison
    if (abs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
      std::cerr << "Mismatch at index " << i << ": CPU " << h_a[i] + h_b[i]
                << " != GPU " << h_c[i] << std::endl;
      correct = false;
      break;
    }
  }
  if (correct) {
    std::cout << "Verification: SUCCESS! GPU results match CPU." << std::endl;
  } else {
    std::cout << "Verification: FAILED!" << std::endl;
  }

  // Cleanup
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Done!" << std::endl;

  return 0;
}
