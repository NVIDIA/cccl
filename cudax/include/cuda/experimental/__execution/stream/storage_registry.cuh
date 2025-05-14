//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__EXECUTION_STORAGE_REGISTRY
#define __CUDAX__EXECUTION_STORAGE_REGISTRY

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_trivially_destructible.h>
#include <cuda/std/cstddef>
#include <cuda/std/span>

#include <cuda/experimental/__algorithm/copy.cuh>
#include <cuda/experimental/__container/uninitialized_async_buffer.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__memory_resource/managed_memory_resource.cuh>

#include <vector>

#include <cuda/experimental/__execution/prologue.cuh>

// The storage registry is a slab allocator of managed memory. It is used by the CUDA
// stream scheduler to allocate temporary storage that will be used by the senders that
// are scheduled on the stream. When "compiling" a sender expression into an operation
// state, each sender in the tree can ask the registry to reserve slots of memory suitable
// to store some type(s), and receive tokens in return. The total memory is allocated in
// one chunk, with room to store the operation state and some metadata as well. During
// execution, each sender's operation state can access the memory it reserved by using the
// token it received. The registry is also responsible for freeing the memory when the
// operation state is destroyed, running any destructors of the types stored in the
// memory.

namespace cuda::experimental::execution
{
struct __storage_registry_context;
using __storage_registry_buffer_t =
  cuda::experimental::uninitialized_async_buffer<_CUDA_VSTD_NOVERSION::byte, mr::host_accessible, mr::device_accessible>;

struct __storage_descriptor
{
  size_t __offset_;
  void (*__destroy_)(_CUDA_VSTD_NOVERSION::byte*) noexcept;
};

_CCCL_API constexpr auto __aligned_size(size_t __size) noexcept -> size_t
{
  // Round the size of _Ty up to the nearest multiple of max_align_t
  constexpr size_t __max_align = alignof(_CUDA_VSTD::max_align_t);
  __size += __max_align - 1;
  __size -= __size % __max_align;
  return __size;
}

/////////////////////////////////////////////////////////////////////////////
// __storage_registry
struct __storage_registry
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Ty>
  _CCCL_API auto __write_at(size_t __token, _Ty __value) -> _Ty&
  {
    constexpr auto __bytes = __aligned_size(sizeof(_Ty));
    auto* __ptr = ::new (__buffer_ + (__descriptors_[__token].__offset_ - __bytes)) _Ty(static_cast<_Ty&&>(__value));
    if constexpr (!::std::is_trivially_destructible_v<_Ty>)
    {
      auto __dtor = +[](_CUDA_VSTD_NOVERSION::byte* __p) noexcept {
        reinterpret_cast<_Ty*>(__p - __bytes)->~_Ty();
      };
      __descriptors_[__token].__destroy_ = __dtor;
    }
    return *__ptr;
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fn, class... _Args>
  _CCCL_API auto __write_at_from(size_t __token, _Fn __fn, _Args&&... __args) -> decltype(auto)
  {
    using _Ty              = __decay_t<decltype(static_cast<_Fn&&>(__fn)(static_cast<_Args&&>(__args)...))>;
    constexpr auto __bytes = __aligned_size(sizeof(_Ty));
    auto* pb               = __buffer_ + (__descriptors_[__token].__offset_ - __bytes);
    auto* __ptr            = ::new (pb) _Ty(static_cast<_Fn&&>(__fn)(static_cast<_Args&&>(__args)...));
    if constexpr (!_CUDA_VSTD::is_trivially_destructible_v<_Ty>)
    {
      auto __dtor = +[](_CUDA_VSTD_NOVERSION::byte* __p) noexcept {
        reinterpret_cast<_Ty*>(__p - __bytes)->~_Ty();
      };
      __descriptors_[__token].__destroy_ = __dtor;
    }
    return *__ptr;
  }

  template <class _Ty>
  _CCCL_API auto __read_at(size_t __token) noexcept -> _Ty&
  {
    constexpr auto __bytes = __aligned_size(sizeof(_Ty));
    auto __offset          = __descriptors_[__token].__offset_ - __bytes;
    return *reinterpret_cast<_Ty*>(__buffer_ + __offset);
  }

  using __descriptor_t                  = __storage_descriptor;
  __descriptor_t* __descriptors_        = nullptr; // points to managed memory
  _CUDA_VSTD_NOVERSION::byte* __buffer_ = nullptr; // points to managed memory
};

/////////////////////////////////////////////////////////////////////////////
// __storage_registry_context
struct __storage_registry_context
{
  _CCCL_HOST_API explicit __storage_registry_context(stream_ref __stream) noexcept
      : __buffer_{experimental::managed_memory_resource{}, __stream, 0}
  {}

  __storage_registry_context(__storage_registry_context&&) noexcept            = delete;
  __storage_registry_context& operator=(__storage_registry_context&&) noexcept = delete;

  _CCCL_HOST_API ~__storage_registry_context() noexcept
  {
    for (auto& __desc : __host_descriptors_)
    {
      if (__desc.__destroy_)
      {
        __desc.__destroy_(__buffer_.data() + __desc.__offset_);
      }
    }
  }

  template <class _Ty>
  _CCCL_HOST_API auto __reserve_for() -> size_t
  {
    return __reserve(__aligned_size(sizeof(_Ty)));
  }

  _CCCL_HOST_API auto __finalize() -> __storage_registry
  {
    if (__host_descriptors_.empty())
    {
      return __storage_registry{nullptr, nullptr};
    }

    // Reserve space for the __descriptors_ to live in the __buffer_ also:
    const auto __bytes = __aligned_size(sizeof(__descriptor_t) * (__host_descriptors_.size() + 1));
    // NB: This will add one element to __host_descriptors_ (hence the +1 above).
    const auto __token = __reserve(__bytes);

    // Allocate the managed memory for all the temporary storage:
    experimental::managed_memory_resource __mr{};
    __buffer_ = __storage_registry_buffer_t{__mr, __buffer_.get_stream(), __host_descriptors_.back().__offset_};

    // Copy the __descriptors_ into the __buffer_:
    auto* __device_descriptors =
      reinterpret_cast<__descriptor_t*>(__buffer_.data() + (__host_descriptors_[__token].__offset_ - __bytes));

    experimental::copy_bytes(
      __buffer_.get_stream(), __host_descriptors_, _CUDA_VSTD::span{__device_descriptors, __host_descriptors_.size()});

    // Wait until the memcpy is done an then return the __storage_registry:
    __buffer_.get_stream().sync();
    return __storage_registry{__device_descriptors, __buffer_.data()};
  }

private:
  using __descriptor_t = __storage_descriptor;

  _CCCL_HOST_API auto __reserve(size_t __bytes) -> size_t
  {
    auto __token = __host_descriptors_.size();
    auto __base  = __token ? __host_descriptors_.back().__offset_ : 0;
    __host_descriptors_.push_back({__base + __bytes, nullptr});
    return __token;
  }

  ::std::vector<__descriptor_t> __host_descriptors_{};
  __storage_registry_buffer_t __buffer_;
};

//////////////////////////////////////////////////////////////////////////////////////////
// queries regarding the stream context's storage registry
struct get_storage_registry_t
{
  template <class _Env>
  _CCCL_API auto operator()(const _Env& __env) const noexcept -> __storage_registry
  {
    return __env.query(*this);
  }
};

_CCCL_GLOBAL_CONSTANT get_storage_registry_t get_storage_registry{};

struct get_storage_registry_context_t
{
  template <class _Env>
  _CCCL_API auto operator()(const _Env& __env) const noexcept -> const __storage_registry_context&
  {
    return __env.query(*this);
  }
};

_CCCL_GLOBAL_CONSTANT get_storage_registry_context_t get_storage_registry_context{};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX__EXECUTION_STORAGE_REGISTRY
