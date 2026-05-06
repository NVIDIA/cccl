//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdio>

#include <cuda_runtime.h>

#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>

static const char* k_source = R"(

#include <initializer_list>
#include <utility>

#include <cuda_runtime.h>

#ifdef _WIN32
#  define EXPORT __declspec(dllexport)
#else
#  define EXPORT __attribute__((visibility("default")))
#endif

__global__ void device_kernel(int* ptr)
{
  ::std::initializer_list<::std::size_t> meow {42ull, 1337ull};
  *ptr = static_cast<int>(::std::move(*meow.begin()));
}

extern "C" EXPORT void host_entry(int* ptr)
{
  device_kernel<<<1, 1>>>(ptr);
}
)";

int main()
{
  // Detect Clang/CUDA configuration from the build environment
  auto config = hostjit::detectDefaultConfig();

  hostjit::JITCompiler compiler(config);
  if (!compiler.compile(k_source))
  {
    std::fprintf(stderr, "HostJIT compilation failed:\n%s\n", compiler.getLastError().c_str());
    return 1;
  }

  auto host_fn = compiler.getFunction<void (*)(int*)>("host_entry");
  if (!host_fn)
  {
    std::fprintf(stderr, "Symbol 'host_entry' not found\n");
    return 1;
  }

  int* d_ptr = nullptr;
  cudaMalloc(&d_ptr, sizeof(int));

  host_fn(d_ptr);
  cudaDeviceSynchronize();

  int result = 0;
  cudaMemcpy(&result, d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_ptr);

  assert(result == 42 && "device kernel did not write expected value");
  std::printf("freestanding compiler test passed (result=%d)\n", result);
  return 0;
}
