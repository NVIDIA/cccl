//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_runtime_api.h>
// cuda_runtime_api needs to come first

#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/utility>
#include <cuda/stream_ref>

#include <new> // IWYU pragma: keep (needed for placement new)

namespace
{
namespace test
{
struct stream : cuda::stream_ref
{
  stream()
      : cuda::stream_ref(::cudaStream_t{})
  {
    ::cudaStream_t stream{};
    _CCCL_TRY_CUDA_API(::cudaStreamCreate, "failed to create a CUDA stream", &stream);
    static_cast<cuda::stream_ref&>(*this) = cuda::stream_ref(stream);
  }

  cuda::stream_ref ref() const noexcept
  {
    return *this;
  }

  void wait() const
  {
    _CCCL_TRY_CUDA_API(::cudaStreamSynchronize, "failed to synchronize a CUDA stream", get());
  }

  ~stream()
  {
    [[maybe_unused]] auto status = ::cudaStreamDestroy(get());
  }
};

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

/// A kernel that takes a callable object and invokes it with a set of arguments
template <class Fn, class... Args>
__global__ void invokernel(Fn fn, Args... args)
{
  fn(args...);
}

} // namespace test
} // namespace
