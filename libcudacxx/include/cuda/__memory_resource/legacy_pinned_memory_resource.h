//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_RESOURCE_LEGACY_PINNED_MEMORY_RESOURCE_H
#define _CUDA___MEMORY_RESOURCE_LEGACY_PINNED_MEMORY_RESOURCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__device/device_ref.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__memory_resource/resource.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__runtime/ensure_current_context.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__exception/throw_error.h>

#  include <cuda/std/__cccl/prologue.h>

//! @file
//! The \c legacy_pinned_memory_resource class provides a memory resource that allocates pinned memory.
_CCCL_BEGIN_NAMESPACE_CUDA_MR

//! @brief legacy_pinned_memory_resource uses `cudaMallocHost` / `cudaFreeAsync` for allocation / deallocation.
//! @note This memory resource will be deprecated in the future. For CUDA 12.6 and above, use
//! `cuda::pinned_memory_resource` instead, which is the long-term replacement.
class legacy_pinned_memory_resource
{
public:
  //! @brief Construct a new legacy_pinned_memory_resource.
  //! @note Synchronous allocations in CUDA are tied to a device, even if not located in device memory.
  //! This constructor takes an optional device argument to specify the device that should be tied to allocations
  //! for the resource. This association has the effect of initializing that device and the memory being implicitly
  //! freed if the device is reset.
  _CCCL_HOST_API constexpr legacy_pinned_memory_resource(::cuda::device_ref __device = {0}) noexcept
      : __device_(__device)
  {}

  //! @brief Allocate host memory of size at least \p __bytes.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @throw std::invalid_argument in case of invalid alignment or \c cuda::cuda_error of the returned error code.
  //! @return Pointer to the newly allocated memory
  [[nodiscard]] _CCCL_HOST_API void*
  allocate_sync(const size_t __bytes, const size_t __alignment = ::cuda::mr::default_cuda_malloc_alignment)
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    if (!__is_valid_alignment(__alignment))
    {
      ::cuda::std::__throw_invalid_argument(
        "Invalid alignment passed to "
        "legacy_pinned_memory_resource::allocate_sync.");
    }

    ::cuda::__ensure_current_context __guard(__device_);
    void* __ptr = ::cuda::__driver::__mallocHost(__bytes);
    return __ptr;
  }

  //! @brief Deallocate memory pointed to by \p __ptr.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate_sync`.
  //! @param __bytes The number of bytes that was passed to the allocation call that returned \p __ptr.
  //! @param __alignment The alignment that was passed to the allocation call that returned \p __ptr.
  _CCCL_HOST_API void deallocate_sync(
    void* __ptr,
    const size_t,
    [[maybe_unused]] const size_t __alignment = ::cuda::mr::default_cuda_malloc_alignment) noexcept
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    _CCCL_ASSERT(__is_valid_alignment(__alignment),
                 "Invalid alignment passed to legacy_pinned_memory_resource::deallocate_sync.");
    _CCCL_ASSERT_CUDA_API(
      ::cuda::__driver::__freeHostNoThrow, "legacy_pinned_memory_resource::deallocate_sync failed", __ptr);
  }

  //! @brief Equality comparison with another \c legacy_pinned_memory_resource.
  //! @param __other The other \c legacy_pinned_memory_resource.
  //! @return Whether both \c legacy_pinned_memory_resource were constructed with the same flags.
  [[nodiscard]] _CCCL_HOST_API constexpr bool operator==(legacy_pinned_memory_resource const&) const noexcept
  {
    return true;
  }
#  if _CCCL_STD_VER <= 2017
  //! @brief Equality comparison with another \c legacy_pinned_memory_resource.
  //! @param __other The other \c legacy_pinned_memory_resource.
  //! @return Whether both \c legacy_pinned_memory_resource were constructed with different flags.
  [[nodiscard]] _CCCL_HOST_API constexpr bool operator!=(legacy_pinned_memory_resource const&) const noexcept
  {
    return false;
  }
#  endif // _CCCL_STD_VER <= 2017

  //! @brief Enables the \c device_accessible property
  _CCCL_HOST_API friend constexpr void
  get_property(legacy_pinned_memory_resource const&, ::cuda::mr::device_accessible) noexcept
  {}
  //! @brief Enables the \c host_accessible property
  _CCCL_HOST_API friend constexpr void
  get_property(legacy_pinned_memory_resource const&, ::cuda::mr::host_accessible) noexcept
  {}

  //! @brief Checks whether the passed in alignment is valid
  _CCCL_HOST_API static constexpr bool __is_valid_alignment(const size_t __alignment) noexcept
  {
    return __alignment <= ::cuda::mr::default_cuda_malloc_alignment
        && (::cuda::mr::default_cuda_malloc_alignment % __alignment == 0);
  }

  using default_queries = ::cuda::mr::properties_list<::cuda::mr::device_accessible, ::cuda::mr::host_accessible>;

private:
  device_ref __device_{0};
};

static_assert(::cuda::mr::synchronous_resource_with<legacy_pinned_memory_resource, ::cuda::mr::device_accessible>, "");
static_assert(::cuda::mr::synchronous_resource_with<legacy_pinned_memory_resource, ::cuda::mr::host_accessible>, "");

_CCCL_END_NAMESPACE_CUDA_MR

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif //_CUDA___MEMORY_RESOURCE_LEGACY_PINNED_MEMORY_RESOURCE_H
