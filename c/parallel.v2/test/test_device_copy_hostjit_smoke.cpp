//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>

#include <cuda_runtime_api.h>

#include <catch2/catch_test_macros.hpp>
#include <hostjit/codegen/cub_call.hpp>
#include <hostjit/jit_compiler.hpp>

namespace
{
template <typename T>
struct device_deleter
{
  void operator()(T* ptr) const
  {
    if (ptr != nullptr)
    {
      cudaFree(ptr);
    }
  }
};

template <typename T>
using device_ptr = std::unique_ptr<T, device_deleter<T>>;

template <typename T>
device_ptr<T> make_device_buffer(std::size_t num_items)
{
  T* ptr = nullptr;
  CATCH_REQUIRE(cudaMalloc(reinterpret_cast<void**>(&ptr), num_items * sizeof(T)) == cudaSuccess);
  return device_ptr<T>{ptr};
}

int current_device_sm()
{
  int device_count               = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status == cudaErrorNoDevice || count_status == cudaErrorInsufficientDriver || device_count == 0)
  {
    CATCH_SKIP("HostJIT DeviceCopy smoke test requires a CUDA device");
  }
  CATCH_REQUIRE(count_status == cudaSuccess);

  int device = 0;
  CATCH_REQUIRE(cudaGetDevice(&device) == cudaSuccess);

  cudaDeviceProp properties{};
  CATCH_REQUIRE(cudaGetDeviceProperties(&properties, device) == cudaSuccess);
  return properties.major * 10 + properties.minor;
}

constexpr std::string_view device_copy_source = R"(
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/cstddef>
#include <cuda/std/mdspan>
#include <cuda/stream_ref>
#include <cub/device/device_copy.cuh>

#if _CCCL_HOSTJIT() && !_CCCL_HOSTED() && !defined(__CUDA_ARCH__) && !defined(_WIN32)
extern "C" void* dlopen(const char*, int);
extern "C" void* dlsym(void*, const char*);
#  ifndef RTLD_NOW
#    define RTLD_NOW 2
#  endif
#endif

static int __cccl_hostjit_init_cuda_driver()
{
  static int status = []() {
#if _CCCL_HOSTJIT() && !_CCCL_HOSTED()
#  if defined(__CUDA_ARCH__)
  return static_cast<int>(cudaSuccess);
#  elif defined(_WIN32)
  return static_cast<int>(cudaErrorNotSupported);
#  else
  if (::cuda::__driver::__getProcAddressFn() != nullptr)
  {
    return static_cast<int>(cudaSuccess);
  }

  static void* driver_library = ::dlopen("libcuda.so.1", RTLD_NOW);
  if (driver_library == nullptr)
  {
    return static_cast<int>(cudaErrorInitializationError);
  }

  static void* get_proc_address = ::dlsym(driver_library, "cuGetProcAddress_v2");
  if (get_proc_address == nullptr)
  {
    return static_cast<int>(cudaErrorInitializationError);
  }

  ::cuda::__driver::__getProcAddressFn(
    reinterpret_cast<decltype(::cuGetProcAddress)*>(get_proc_address),
    true);
#  endif
#endif // _CCCL_HOSTJIT() && !_CCCL_HOSTED()

  return static_cast<int>(cudaSuccess);
  }();
  return status;
}

extern "C" _CCCL_VISIBILITY_EXPORT int cccl_jit_device_copy(
    void* d_temp_storage,
    size_t* temp_storage_bytes,
    void* d_in,
    void* d_out,
    unsigned long long num_items,
    void* stream)
{
  int __cccl_hostjit_driver_status = __cccl_hostjit_init_cuda_driver();
  if (__cccl_hostjit_driver_status != static_cast<int>(cudaSuccess))
  {
    return __cccl_hostjit_driver_status;
  }

  using extents_t = ::cuda::std::extents<::cuda::std::size_t, ::cuda::std::dynamic_extent>;
  using mdspan_t = ::cuda::std::mdspan<int, extents_t>;

  mdspan_t in(static_cast<int*>(d_in), num_items);
  mdspan_t out(static_cast<int*>(d_out), num_items);

  cudaError_t err = cub::DeviceCopy::Copy(
    d_temp_storage,
    *temp_storage_bytes,
    in,
    out,
    ::cuda::std::execution::env{::cuda::stream_ref{static_cast<cudaStream_t>(stream)}});
  return static_cast<int>(err);
}
)";
} // namespace

CATCH_TEST_CASE("HostJIT smoke test can execute cub::DeviceCopy::Copy", "[hostjit][device_copy]")
{
  constexpr std::size_t num_items = 257;
  constexpr std::size_t num_bytes = num_items * sizeof(int);

  using copy_fn_t = int (*)(void*, std::size_t*, void*, void*, unsigned long long, void*);

  const int sm_version = current_device_sm();
  auto config          = hostjit::codegen::CubCall::make_jit_config(
    sm_version / 10, sm_version % 10, nullptr, nullptr, nullptr, "cccl_jit_device_copy");

  hostjit::JITCompiler compiler(config);
  const bool compiled = compiler.compile(std::string{device_copy_source});
  CATCH_INFO(compiler.getLastError());
  CATCH_REQUIRE(compiled);

  copy_fn_t device_copy = compiler.getFunction<copy_fn_t>("cccl_jit_device_copy");
  CATCH_INFO(compiler.getLastError());
  CATCH_REQUIRE(device_copy != nullptr);

  std::array<int, num_items> host_input{};
  std::iota(host_input.begin(), host_input.end(), 42);

  std::array<int, num_items> host_output{};
  std::fill(host_output.begin(), host_output.end(), -1);

  auto d_input  = make_device_buffer<int>(num_items);
  auto d_output = make_device_buffer<int>(num_items);

  CATCH_REQUIRE(cudaMemcpy(d_input.get(), host_input.data(), num_bytes, cudaMemcpyHostToDevice) == cudaSuccess);
  CATCH_REQUIRE(cudaMemset(d_output.get(), 0, num_bytes) == cudaSuccess);

  std::size_t temp_storage_bytes = 0;
  CATCH_REQUIRE(
    device_copy(nullptr, &temp_storage_bytes, d_input.get(), d_output.get(), num_items, nullptr) == cudaSuccess);
  CATCH_REQUIRE(temp_storage_bytes != 0);

  auto d_temp_storage = make_device_buffer<std::byte>(temp_storage_bytes);
  CATCH_REQUIRE(
    device_copy(d_temp_storage.get(), &temp_storage_bytes, d_input.get(), d_output.get(), num_items, nullptr)
    == cudaSuccess);
  CATCH_REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  CATCH_REQUIRE(cudaMemcpy(host_output.data(), d_output.get(), num_bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
  CATCH_REQUIRE(host_output == host_input);
}
