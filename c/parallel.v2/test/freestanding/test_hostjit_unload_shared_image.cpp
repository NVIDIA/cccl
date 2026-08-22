//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// One module image, two handles. Loading the same library twice gives
// two handles to ONE mapped image, and the fatbin-unregister callbacks live in
// that image, shared. Unloading one handle therefore unregisters the fatbin out
// from under the other handle, which is still open and still callable.
//
// This does not check a requirement, it pins what happens today: registration in
// CUDART is one-per-register with no reference count, so the loader cannot let
// the last holder decide when to unregister without counting holders itself.
// When that counting arrives, this test fails and says so, instead of the change
// going unnoticed.
//
// Both platforms behave the same way here, because the produced DLL names its own
// entry point and the OS therefore runs the static initializers on the first load
// only, registering the fatbin once, as dlopen does.

#include <cstdio>
#include <filesystem>
#include <string>

#include <cuda_runtime.h>

#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>
#include <hostjit/loader.hpp>

namespace
{
const char* k_source = R"(
#include <cuda_runtime.h>
#include <cuda/std/version>

__global__ void device_kernel(int* ptr, int v)
{
  *ptr = v;
}

extern "C" _CCCL_VISIBILITY_EXPORT void host_entry(int* ptr, int v)
{
  device_kernel<<<1, 1>>>(ptr, v);
}
)";

// Build the module and keep it on disk, so the probe can open it twice itself.
bool build_module(hostjit::CompilerConfig config, std::string& module_path)
{
  config.enable_pch     = false;
  config.keep_artifacts = true;

  hostjit::JITCompiler compiler(config);
  if (!compiler.compile(k_source))
  {
    std::fprintf(stderr, "  compile failed: %s\n", compiler.getLastError().c_str());
    return false;
  }
  module_path = compiler.getLoadedModulePath();
  return !module_path.empty() && std::filesystem::exists(module_path);
}

// Launch through one handle and report the outcome instead of asserting it:
// this probe is about what the runtime does, not about a known-good result.
cudaError_t launch_through(hostjit::DynamicLibrary& lib, int* d_ptr, int v)
{
  auto host_entry = reinterpret_cast<void (*)(int*, int)>(lib.getSymbol("host_entry"));
  if (!host_entry)
  {
    return cudaErrorSymbolNotFound;
  }
  cudaGetLastError(); // clear anything pending
  host_entry(d_ptr, v);
  const cudaError_t launch = cudaGetLastError();
  const cudaError_t sync   = cudaDeviceSynchronize();
  return launch != cudaSuccess ? launch : sync;
}

bool probe_shared_image(const std::string& module_path, int* d_ptr)
{
  hostjit::DynamicLibrary first, second;
  if (!first.load(module_path) || !second.load(module_path))
  {
    std::fprintf(stderr, "  could not open the module twice\n");
    return false;
  }

  const cudaError_t before_first  = launch_through(first, d_ptr, 1);
  const cudaError_t before_second = launch_through(second, d_ptr, 2);
  std::printf(
    "  both handles open: first=%s second=%s\n", cudaGetErrorName(before_first), cudaGetErrorName(before_second));
  if (before_first != cudaSuccess || before_second != cudaSuccess)
  {
    std::fprintf(stderr, "  a launch failed while both handles were open\n");
    return false;
  }

  first.unload();

  const cudaError_t after = launch_through(second, d_ptr, 3);
  std::printf("  after unloading the first handle, the second launches: %s\n", cudaGetErrorName(after));

  second.unload();
  cudaGetLastError();

  // Today the fatbin is gone while the second handle is still open, so the
  // launch fails. If this ever starts succeeding -- because registration became
  // reference-counted, or because each handle got its own image -- the probe
  // fails and this file should be updated.
  const bool as_documented = after != cudaSuccess;
  if (!as_documented)
  {
    std::fprintf(stderr, "  the second handle still works: unload is no longer image-wide -- update the notes\n");
  }
  return as_documented;
}
} // namespace

int main()
{
  std::printf("unload-shared-image -- one module image held by two handles\n");

  auto config = hostjit::detectDefaultConfig();

  int* d_ptr = nullptr;
  if (cudaMalloc(&d_ptr, sizeof(int)) != cudaSuccess)
  {
    std::fprintf(stderr, "unload-shared-image: cudaMalloc failed\n");
    return 2;
  }

  std::string module_path;
  if (!build_module(config, module_path))
  {
    cudaFree(d_ptr);
    return 2;
  }

  const bool ok = probe_shared_image(module_path, d_ptr);
  cudaFree(d_ptr);

  std::printf("unload-shared-image: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
