#include <iostream>
#include <numeric>

#include <cuda_runtime.h>

#include "test_util.h"
#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>

static const char* cuda_source = R"(
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <cuda/std/functional>
#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

#ifdef _WIN32
#define HOSTJIT_EXPORT __declspec(dllexport)
#else
#define HOSTJIT_EXPORT
#endif

extern "C" HOSTJIT_EXPORT int computeSumDeterministic(float* d_input, int num_items, float* result) {
    float* d_output = nullptr;

    // Allocate output
    cudaError_t err = cudaMalloc(&d_output, sizeof(float));
    if (err != cudaSuccess) return -1;

    // Run deterministic sum reduction with gpu_to_gpu determinism
    auto env = cuda::execution::require(cuda::execution::determinism::gpu_to_gpu);
    err = cub::DeviceReduce::Sum(d_input, d_output, num_items, env);
    if (err != cudaSuccess) {
        cudaFree(d_output);
        return -2;
    }

    // Synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_output);
        return -3;
    }

    // Copy result back
    err = cudaMemcpy(result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_output);
        return -4;
    }

    // Cleanup
    cudaFree(d_output);

    return 0;
}
)";

int main()
{
  // Detect Clang/CUDA configuration from the build environment
  auto config = hostjit::detectDefaultConfig();

  hostjit::JITCompiler compiler(config);
  if (!compiler.compile(cuda_source))
  {
    std::fprintf(stderr, "HostJIT compilation failed:\n%s\n", compiler.getLastError().c_str());
    return 1;
  }

  auto computeSumDeterministic = compiler.getFunction<int (*)(float*, int, float*)>("computeSumDeterministic");
  if (!computeSumDeterministic)
  {
    std::cerr << "Failed to get function: " << compiler.getLastError() << "\n";
    return 1;
  }

  // Prepare test data
  const int N = 1024;
  std::vector<float> h_input(N);

  // Initialize with values 1.0 to N
  std::iota(h_input.begin(), h_input.end(), 1.0f);

  // Expected sum: N*(N+1)/2
  float expected_sum = static_cast<float>(N) * (N + 1) / 2;

  // Allocate device memory
  float* d_input;
  CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  // Call the JIT-compiled function
  std::cout << "Computing deterministic sum of " << N << " floats using CUB...\n";
  float result = 0;
  int status   = computeSumDeterministic(d_input, N, &result);

  if (status != 0)
  {
    std::cerr << "computeSumDeterministic failed with status: " << status << "\n";
    CUDA_CHECK(cudaFree(d_input));
    return 1;
  }

  // Verify result
  std::cout << "Result: " << result << " (expected: " << expected_sum << ")\n";

  // Verify determinism: run multiple times and check results are identical
  const int num_runs = 10;
  bool deterministic = true;
  for (int i = 0; i < num_runs; ++i)
  {
    float run_result = 0;
    status           = computeSumDeterministic(d_input, N, &run_result);
    if (status != 0)
    {
      std::cerr << "Run " << i << " failed with status: " << status << "\n";
      CUDA_CHECK(cudaFree(d_input));
      return 1;
    }
    if (run_result != result)
    {
      std::cerr << "Run " << i << " produced different result: " << run_result << " vs " << result << "\n";
      deterministic = false;
    }
  }

  bool success = (result == expected_sum) && deterministic;
  if (success)
  {
    std::cout << "Results verified successfully! (" << num_runs << " runs produced identical results)\n";
  }
  else
  {
    if (result != expected_sum)
    {
      std::cerr << "Mismatch! Got " << result << " but expected " << expected_sum << "\n";
    }
    if (!deterministic)
    {
      std::cerr << "Non-deterministic results detected!\n";
    }
  }

  // Cleanup
  CUDA_CHECK(cudaFree(d_input));

  return success ? 0 : 1;
}
