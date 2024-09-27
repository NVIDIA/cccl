//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __COMMON_UTILITY_H__
#define __COMMON_UTILITY_H__

#include <cuda_runtime_api.h>
// cuda_runtime_api needs to come first

#include <cuda/atomic>
#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/utility>
#include <cuda/stream_ref>

#include <cuda/experimental/launch.cuh>

#include <new> // IWYU pragma: keep (needed for placement new)

#include "testing.cuh"

namespace
{
namespace test
{

constexpr auto one_thread_dims = cudax::make_hierarchy(cudax::block_dims<1>(), cudax::grid_dims<1>());

struct _malloc_managed
{
private:
  void* pv = nullptr;

public:
  explicit _malloc_managed(std::size_t size)
  {
    _CCCL_TRY_CUDA_API(::cudaMallocManaged, "failed to allocate managed memory", &pv, size);
  }

  ~_malloc_managed()
  {
    [[maybe_unused]] auto status = ::cudaFree(pv);
  }

  template <class T>
  T* get_as() const noexcept
  {
    return static_cast<T*>(pv);
  }
};

template <class T>
struct managed
{
private:
  _malloc_managed _mem;

public:
  explicit managed(T t)
      : _mem(sizeof(T))
  {
    ::new (_mem.get_as<void>()) T(_CUDA_VSTD::move(t));
  }

  ~managed()
  {
    get()->~T();
  }

  T* get() noexcept
  {
    return _mem.get_as<T>();
  }
  const T* get() const noexcept
  {
    return _mem.get_as<T>();
  }

  T& operator*() noexcept
  {
    return *get();
  }
  const T& operator*() const noexcept
  {
    return *get();
  }
};

struct assign_42
{
  __device__ constexpr void operator()(int* pi) const noexcept
  {
    *pi = 42;
  }
};

struct verify_42
{
  __device__ void operator()(int* pi) const noexcept
  {
    CUDAX_REQUIRE(*pi == 42);
  }
};

struct spin_until_80
{
  __device__ void operator()(int* pi) const noexcept
  {
    cuda::atomic_ref atomic_pi(*pi);
    while (atomic_pi.load() != 80)
      ;
  }
};

struct empty_kernel
{
  __device__ void operator()() const noexcept {}
};

inline int count_driver_stack()
{
  if (cudax::detail::driver::ctxGetCurrent() != nullptr)
  {
    auto ctx    = cudax::detail::driver::ctxPop();
    auto result = 1 + count_driver_stack();
    cudax::detail::driver::ctxPush(ctx);
    return result;
  }
  else
  {
    return 0;
  }
}

inline void empty_driver_stack()
{
  while (cudax::detail::driver::ctxGetCurrent() != nullptr)
  {
    cudax::detail::driver::ctxPop();
  }
}

inline int cuda_driver_version()
{
  return cudax::detail::driver::getVersion();
}

} // namespace test
} // namespace
#endif // __COMMON_UTILITY_H__
