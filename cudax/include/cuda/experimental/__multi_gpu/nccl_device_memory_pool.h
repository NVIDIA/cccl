//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_NCCL_DEVICE_MEMORY_POOL_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_NCCL_DEVICE_MEMORY_POOL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/attributes.h>
#include <cuda/__device/device_ref.h>
#include <cuda/__memory_pool/device_memory_pool.h>
#include <cuda/__memory_pool/memory_pool_base.h>
#include <cuda/std/__host_stdlib/mutex>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__memory/unique_ptr.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
class __default_nccl_device_memory_pool
{
public:
  [[nodiscard]] _CCCL_HOST_API ::cuda::device_memory_pool_ref& __get(::cuda::device_ref __device)
  {
#if _CCCL_HOSTED()
    ::std::call_once(__once_, [this, __device] {
      this->__init(__device);
    });
#else // ^^^ _CCCL_HOSTED() ^^^ / vvv _CCCL_FREESTANDING() vvv
    if (!__initialized_)
    {
      this->__init(__device);
      __initialized_ = true;
    }
#endif // _CCCL_FREESTANDING()
    return __storage_.__pool_;
  }

private:
  _CCCL_HOST_API void __init(::cuda::device_ref __device)
  {
    auto __props           = ::cuda::memory_pool_properties{};
    using __alloc_handle_t = ::cuda::device_attributes::memory_pool_supported_handle_types_t;

    __props.allocation_handle_type = __alloc_handle_t::posix_file_descriptor;

    // Though we cna name __alloc_handle_t::fabric regardless of CUDA version, we cannot query
    // it without error before 12.4.
#if _CCCL_CTK_AT_LEAST(12, 4)
    if (::cuda::device_attributes::memory_pool_supported_handle_types(__device) & __alloc_handle_t::fabric)
    {
      __props.allocation_handle_type = static_cast<decltype(__props.allocation_handle_type)>(
        __props.allocation_handle_type | __alloc_handle_t::fabric);
    }
#endif

    auto __loc = ::CUmemLocation{};

    __loc.type = ::CU_MEM_LOCATION_TYPE_DEVICE;
    __loc.id   = __device.get();

    // Hold in an owned pool because enable_access_from() may throw
    auto __pool = ::cuda::device_memory_pool::from_native_handle(
      ::cuda::__create_cuda_mempool(__props, __loc, ::CU_MEM_ALLOCATION_TYPE_PINNED));

    __pool.enable_access_from(__device.peers());

    ::cuda::std::__construct_at(&__storage_.__pool_, __pool.get());
    // Only after we have "transferred" ownership to the ref is it safe to release. Technically
    // we could release in the construct_at() call as well since the ref doesn't actually own
    // anything and won't clean up anyway.
    static_cast<void>(__pool.release());
  }

#if _CCCL_HOSTED()
  ::std::once_flag __once_{};
#else // ^^^ _CCCL_HOSTED() ^^^ / vvv _CCCL_FREESTANDING() vvv
  bool __initialized_{false};
#endif // _CCCL_FREESTANDING()

  union __storage_t
  {
    char __empty_;
    ::cuda::device_memory_pool_ref __pool_;

    _CCCL_HOST_API __storage_t() noexcept
        : __empty_{}
    {}
  };

  __storage_t __storage_;
};

//! @brief Returns the default NCCL compatible memory pool for a device.
//!
//! Memory allocated from the returned pool has the same properties as memory returned by
//! `ncclMemAlloc()`, so it may be registered with a NCCL communicator as a user buffer. See
//! https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html for the
//! buffer registration requirements. Allocations are also accessible from every peer of
//! `__device`.
//!
//! There is exactly one pool per device, so every call with the same `__device` returns a
//! reference to the same pool. The pool remains valid for the duration of the program.
//!
//! `__device` must refer to a device that supports exportable memory pools, otherwise an
//! exception is thrown.
//!
//! @note This function is thread safe when the host standard library is available.
//!
//! @snippet nccl/nccl_device_memory_pool.cu device_default_nccl_memory_pool_allocate
//
//! @param __device The device whose default NCCL memory pool is returned.
//!
//! @returns A reference to the default NCCL memory pool for `__device`.
//!
//! @throws cuda_error If the memory pool could not be created.
[[nodiscard]] _CCCL_HOST_API inline ::cuda::device_memory_pool_ref&
device_default_nccl_memory_pool(::cuda::device_ref __device)
{
  static ::cuda::std::unique_ptr<__default_nccl_device_memory_pool[]> __pools_{
    ::new __default_nccl_device_memory_pool[::cuda::__physical_devices_count()]};

  return __pools_[static_cast<::cuda::std::size_t>(__device.get())].__get(__device);
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

// NOLINTEND(bugprone-reserved-identifier)

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_NCCL_DEVICE_MEMORY_POOL_H
