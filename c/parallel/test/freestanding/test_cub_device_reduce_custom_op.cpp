#include <iostream>
#include <numeric>

#include <cuda_runtime.h>

#include "test_util.h"
#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>

static const char* cuda_source = R"(
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>

// External declaration - resolved from linked bitcode
extern "C" __device__ float user_op(float a, float b);

// Functor wrapping the external function
struct UserOp {
    __device__ __forceinline__
    float operator()(float a, float b) const {
        return ::cuda::std::min(a, b);
    }
};

#ifdef _WIN32
#define HOSTJIT_EXPORT __declspec(dllexport)
#else
#define HOSTJIT_EXPORT
#endif

extern "C" HOSTJIT_EXPORT int computeReduce(float* d_input, int num_items, float* result) {
    float* d_output = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    float init = 1e38f;
    UserOp op;

    cudaError_t err = cudaMalloc(&d_output, sizeof(float));
    if (err != cudaSuccess) return -1;

    err = cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes,
                                     d_input, d_output, num_items, op, init);
    if (err != cudaSuccess) {
        cudaFree(d_output);
        return -2;
    }

    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_output);
        return -3;
    }

    err = cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes,
                                     d_input, d_output, num_items, op, init);
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_output);
        return -4;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        cudaFree(d_output);
        return -5;
    }

    err = cudaMemcpy(result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_temp_storage);
    cudaFree(d_output);

    return (err == cudaSuccess) ? 0 : -6;
}
)";

int main()
{
  // Detect Clang/CUDA configuration from the build environment
  auto config = hostjit::detectDefaultConfig();

  hostjit::JITCompiler compiler(config);
  if (!compiler.compile(cuda_source))
  {
    std::cerr << "Failed to get function: " << compiler.getLastError() << "\n";
    return 1;
  }

  auto computeReduce = compiler.getFunction<int (*)(float*, int, float*)>("computeReduce");
  if (!computeReduce)
  {
    std::cerr << "Failed to get function: " << compiler.getLastError() << "\n";
    return 1;
  }

  // Prepare test data
  const int N = 1024;
  std::vector<float> h_input(N);

  // Initialize with values 1.0 to N
  std::iota(h_input.begin(), h_input.end(), 1.0f);

  const float expected = 1.0f; // Minimum value in the array

  // Allocate device memory
  float* d_input;
  CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  float result = -1.0f;
  int status   = computeReduce(d_input, N, &result);
  if (status != 0)
  {
    std::cerr << "computeReduce failed with status: " << status << "\n";
    CUDA_CHECK(cudaFree(d_input));
    return 1;
  }

  // Verify result
  std::cout << "\nResult: " << result << " (expected: " << expected << ")\n";
  bool success = std::abs(result - expected) < 0.01f;
  if (success)
  {
    std::cout << "\n*** SUCCESS: Custom operator was inlined and executed correctly! ***\n";
  }
  else
  {
    std::cerr << "\nMismatch! Got " << result << " but expected " << expected << "\n";
  }

  // Cleanup
  CUDA_CHECK(cudaFree(d_input));

  return success ? 0 : 1;
}
