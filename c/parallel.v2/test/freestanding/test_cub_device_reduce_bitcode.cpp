#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

#include <cuda_runtime.h>

#include "test_util.h"
#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>

// LLVM IR for an add operator with alwaysinline attribute
// This IR defines a function: float user_op(float a, float b) { return a + b; }
const char* user_op_llvm_ir = R"(
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define float @user_op(float %a, float %b) alwaysinline {
entry:
  %result = fadd float %a, %b
  ret float %result
}
)";

// Write LLVM IR text to a file (parseIRFile in the compiler handles both .ll and .bc)
bool writeIRFile(const std::string& llvm_ir, const std::string& output_path)
{
  std::ofstream ir_file(output_path);
  if (!ir_file)
  {
    std::cerr << "Failed to write LLVM IR file\n";
    return false;
  }
  ir_file << llvm_ir;
  ir_file.close();
  std::cout << "Generated IR file: " << output_path << "\n";
  return true;
}

static const char* cuda_source = R"(
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>

// External declaration - resolved from linked bitcode
extern "C" __device__ float user_op(float a, float b);

// Functor wrapping the external function
struct UserOp {
    __device__ __forceinline__
    float operator()(float a, float b) const {
        return user_op(a, b);
    }
};

extern "C" _CCCL_VISIBILITY_EXPORT int computeReduce(float* d_input, int num_items, float* result) {
    float* d_output = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    float init = HOSTJIT_REDUCE_INIT;
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
  // Step 1: Write LLVM IR file with the custom operator
  std::string ir_path = (std::filesystem::temp_directory_path() / "user_op.ll").string();
  if (!writeIRFile(user_op_llvm_ir, ir_path))
  {
    std::cerr << "Failed to write IR file\n";
    return 1;
  }

  auto config = hostjit::detectDefaultConfig();
  config.device_bitcode_files.push_back(ir_path);
  config.macro_definitions["HOSTJIT_REDUCE_INIT"] = "0.0f";

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
  const float expected = static_cast<float>(N) * (N + 1) / 2.0f; // Sum of 1..N

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
