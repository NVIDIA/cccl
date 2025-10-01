//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_LEGACY_MANAGED_MEMORY_RESOURCE_CUH
#define _CUDAX__MEMORY_RESOURCE_LEGACY_MANAGED_MEMORY_RESOURCE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILER(CLANG)
#  include <cuda_runtime_api.h>
#endif // _CCCL_CUDA_COMPILER(CLANG)

#include <cuda/__memory_resource/get_property.h>
#include <cuda/__memory_resource/properties.h>
#include <cuda/__memory_resource/resource.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/detail/libcxx/include/stdexcept>

#include <cuda/experimental/__memory_resource/any_resource.cuh>
#include <cuda/experimental/__memory_resource/properties.cuh>

#include <cuda/std/__cccl/prologue.h>

//! @file
//! The \c managed_memory_resource class provides a memory resource that allocates managed memory.
namespace cuda::experimental
{

//! @brief \c managed_memory_resource uses `cudaMallocManaged` / `cudaFree` for allocation / deallocation.
class legacy_managed_memory_resource
{
private:
  unsigned int __flags_ = cudaMemAttachGlobal;

  static constexpr unsigned int __available_flags = cudaMemAttachGlobal | cudaMemAttachHost;

public:
  constexpr legacy_managed_memory_resource(const unsigned int __flags = cudaMemAttachGlobal) noexcept
      : __flags_(__flags & __available_flags)
  {
    _CCCL_ASSERT(__flags_ == __flags, "Unexpected flags passed to legacy_managed_memory_resource");
  }

  //! @brief Allocate CUDA unified memory of size at least \p __bytes.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @throw std::invalid_argument in case of invalid alignment or \c cuda::cuda_error of the returned error code.
  //! @return Pointer to the newly allocated memory
  [[nodiscard]] void* allocate_sync(const size_t __bytes,
                                    const size_t __alignment = ::cuda::mr::default_cuda_malloc_alignment) const
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    if (!__is_valid_alignment(__alignment))
    {
      ::cuda::std::__throw_invalid_argument(
        "Invalid alignment passed to "
        "legacy_managed_memory_resource::allocate_sync.");
    }

    void* __ptr{nullptr};
    _CCCL_TRY_CUDA_API(
      ::cudaMallocManaged, "Failed to allocate memory with cudaMallocManaged.", &__ptr, __bytes, __flags_);
    return __ptr;
  }

  //! @brief Deallocate memory pointed to by \p __ptr.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate` or `allocate_sync`
  //! @param __bytes The number of bytes that was passed to the allocation call that returned \p __ptr.
  //! @param __alignment The alignment that was passed to the allocation call that returned \p __ptr.
  void deallocate_sync(
    void* __ptr,
    const size_t,
    [[maybe_unused]] const size_t __alignment = ::cuda::mr::default_cuda_malloc_alignment) const noexcept
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    _CCCL_ASSERT(__is_valid_alignment(__alignment),
                 "Invalid alignment passed to legacy_managed_memory_resource::deallocate_sync.");
    _CCCL_ASSERT_CUDA_API(::cudaFree, "legacy_managed_memory_resource::deallocate_sync failed", __ptr);
  }

  //! @brief Equality comparison with another \c managed_memory_resource.
  //! @param __other The other \c managed_memory_resource.
  //! @return Whether both \c managed_memory_resource were constructed with the same flags.
  [[nodiscard]] constexpr bool operator==(legacy_managed_memory_resource const& __other) const noexcept
  {
    return __flags_ == __other.__flags_;
  }
#if _CCCL_STD_VER <= 2017
  //! @brief Inequality comparison with another \c managed_memory_resource.
  //! @param __other The other \c managed_memory_resource.
  //! @return Whether both \c managed_memory_resource were constructed with different flags.
  [[nodiscard]] constexpr bool operator!=(legacy_managed_memory_resource const& __other) const noexcept
  {
    return __flags_ != __other.__flags_;
  }
#endif // _CCCL_STD_VER <= 2017

  //! @brief Enables the \c device_accessible property
  friend constexpr void get_property(legacy_managed_memory_resource const&, device_accessible) noexcept {}
  //! @brief Enables the \c host_accessible property
  friend constexpr void get_property(legacy_managed_memory_resource const&, host_accessible) noexcept {}

  //! @brief Checks whether the passed in alignment is valid
  static constexpr bool __is_valid_alignment(const size_t __alignment) noexcept
  {
    return __alignment <= ::cuda::mr::default_cuda_malloc_alignment
        && (::cuda::mr::default_cuda_malloc_alignment % __alignment == 0);
  }

  using default_queries = properties_list<device_accessible, host_accessible>;
};
static_assert(::cuda::mr::synchronous_resource_with<legacy_managed_memory_resource, device_accessible>, "");
static_assert(::cuda::mr::synchronous_resource_with<legacy_managed_memory_resource, host_accessible>, "");

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //_CUDAX__MEMORY_RESOURCE_LEGACY_MANAGED_MEMORY_RESOURCE_CUH
