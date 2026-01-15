#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define N 10000000 // 10 Million

// CUDA Kernel: Sum Array using Grid-Stride Loop + AtomicAdd
__global__ void sumKernel(float *d_a, float *d_sum, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  float local_sum = 0.0f;

  // Grid-Stride Loop
  for (int i = idx; i < n; i += stride) {
    local_sum += d_a[i];
  }

  // Atomic Add to global sum
  atomicAdd(d_sum, local_sum);
}

// CPU Sum Function
float sumArrayCPU(float *h_a, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += h_a[i];
  }
  return sum;
}

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

// Helper to run a test case
float runGpuTest(float *d_a, float *d_sum, int n, int threads, int blocks,
                 cudaEvent_t start, cudaEvent_t stop) {
  float zero = 0.0f;
  checkCudaError(
      cudaMemcpy(d_sum, &zero, sizeof(float), cudaMemcpyHostToDevice),
      "Reset d_sum");

  checkCudaError(cudaEventRecord(start), "Record Start");

  sumKernel<<<blocks, threads>>>(d_a, d_sum, n);

  checkCudaError(cudaEventRecord(stop), "Record Stop");
  checkCudaError(cudaEventSynchronize(stop), "Sync");
  checkCudaError(cudaGetLastError(), "Kernel Launch");

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  return milliseconds;
}

int main() {
  size_t size = N * sizeof(float);

  // Host Memory
  float *h_a = (float *)malloc(size);
  if (h_a == NULL) {
    std::cerr << "Failed to allocate host memory" << std::endl;
    return -1;
  }

  // Initialize with random values
  std::cout << "Initializing Array (N=" << N << ")..." << std::endl;
  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_a[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // 1. CPU Timing
  std::cout << "Running CPU Sum..." << std::endl;
  clock_t cpu_start = clock();
  float cpu_sum = sumArrayCPU(h_a, N);
  clock_t cpu_end = clock();
  double cpu_time_ms = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
  std::cout << "CPU Time: " << cpu_time_ms << " ms | Sum: " << cpu_sum
            << std::endl;
  std::cout << "--------------------------------------------------------"
            << std::endl;

  // Device Memory
  float *d_a, *d_sum;
  checkCudaError(cudaMalloc((void **)&d_a, size), "Alloc d_a");
  checkCudaError(cudaMalloc((void **)&d_sum, sizeof(float)), "Alloc d_sum");

  // Copy to Device
  std::cout << "Copying data to GPU..." << std::endl;
  checkCudaError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice),
                 "Copy HtoD");

  // Start Testing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup
  runGpuTest(d_a, d_sum, N, 128, 128, start, stop);

  std::cout << "Starting GPU Experiments..." << std::endl;
  std::cout << "--------------------------------------------------------"
            << std::endl;

  // Experiment 1: Specific Cases from Lab Sheet
  struct Config {
    int t;
    int b;
    const char *desc;
  };
  std::vector<Config> scenarios = {{128, 256, "Baseline"},
                                   {256, 512, "Question 1"},
                                   {1024, 512, "Question 2"}};

  float gpu_sum_val = 0.0f;

  for (const auto &cfg : scenarios) {
    float time_ms = runGpuTest(d_a, d_sum, N, cfg.t, cfg.b, start, stop);
    checkCudaError(
        cudaMemcpy(&gpu_sum_val, d_sum, sizeof(float), cudaMemcpyDeviceToHost),
        "Get Result");
    std::cout << "[" << cfg.desc << "] Threads: " << cfg.t
              << " | Blocks: " << cfg.b << " | Time: " << time_ms << " ms"
              << std::endl;
  }
  std::cout << "--------------------------------------------------------"
            << std::endl;

  // Experiment 2: Question 3 - Sweep Blocks with Threads=512
  std::cout << "[Question 3] Sweep Blocks (Threads=512):" << std::endl;
  int blockSweep[] = {128, 256, 512, 1024, 2048};
  for (int b : blockSweep) {
    float time_ms = runGpuTest(d_a, d_sum, N, 512, b, start, stop);
    std::cout << "  > Threads: 512 | Blocks: " << b << " | Time: " << time_ms
              << " ms" << std::endl;
  }

  // Verification (Check last GPU result)
  checkCudaError(
      cudaMemcpy(&gpu_sum_val, d_sum, sizeof(float), cudaMemcpyDeviceToHost),
      "Get Final Result");
  float diff = fabs(cpu_sum - gpu_sum_val);
  float pct_diff = diff / cpu_sum * 100.0f;
  std::cout << "--------------------------------------------------------"
            << std::endl;
  std::cout << "Verification:" << std::endl;
  std::cout << "  CPU Sum: " << cpu_sum << std::endl;
  std::cout << "  GPU Sum: " << gpu_sum_val << std::endl;
  std::cout << "  Diff: " << diff << " (" << pct_diff << "%)" << std::endl;

  if (pct_diff <
      0.1) { // Allow slight floating point error due to order of operations
    std::cout << "  Result: SUCCESS (Matches)" << std::endl;
  } else {
    std::cout << "  Result: WARNING (Diff > 0.1%) - Likely due to floating "
                 "point accumulation order"
              << std::endl;
  }

  // Cleanup
  cudaFree(d_a);
  cudaFree(d_sum);
  free(h_a);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
