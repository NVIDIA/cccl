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

constexpr auto one_thread_dims = cudax::make_config(cudax::block_dims<1>(), cudax::grid_dims<1>());

struct _malloc_managed
{
private:
  void* pv = nullptr;

public:
  explicit _malloc_managed(std::size_t size)
  {
    cudax::__ensure_current_device guard(cudax::device_ref{0});
    _CCCL_TRY_CUDA_API(::cudaMallocManaged, "failed to allocate managed memory", &pv, size);
  }

  ~_malloc_managed()
  {
    cudax::__ensure_current_device guard(cudax::device_ref{0});
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

template <int N>
struct assign_n
{
  __device__ constexpr void operator()(int* pi) const noexcept
  {
    *pi = N;
  }
};

template <int N>
struct verify_n
{
  __device__ void operator()(int* pi) const noexcept
  {
    CUDAX_REQUIRE(*pi == N);
  }
};

using assign_42 = assign_n<42>;
using verify_42 = verify_n<42>;

struct atomic_add_one
{
  __device__ void operator()(int* pi) const noexcept
  {
    cuda::atomic_ref atomic_pi(*pi);
    atomic_pi.fetch_add(1);
  }
};

struct atomic_sub_one
{
  __device__ void operator()(int* pi) const noexcept
  {
    cuda::atomic_ref atomic_pi(*pi);
    atomic_pi.fetch_sub(1);
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

} // namespace test
} // namespace
#endif // __COMMON_UTILITY_H__
