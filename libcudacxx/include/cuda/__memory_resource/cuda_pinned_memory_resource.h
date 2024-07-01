//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MEMORY_RESOURCE_CUDA_PINNED_MEMORY_RESOURCE_H
#define _CUDA__MEMORY_RESOURCE_CUDA_PINNED_MEMORY_RESOURCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !defined(_CCCL_COMPILER_MSVC_2017) && defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

#  if !defined(_CCCL_CUDA_COMPILER_NVCC) && !defined(_CCCL_CUDA_COMPILER_NVHPC)
#    include <cuda_runtime.h>
#    include <cuda_runtime_api.h>
#  endif // !_CCCL_CUDA_COMPILER_NVCC && !_CCCL_CUDA_COMPILER_NVHPC

#  include <cuda/__memory_resource/get_property.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__memory_resource/resource.h>
#  include <cuda/__memory_resource/resource_ref.h>
#  include <cuda/std/__concepts/__concept_macros.h>
#  include <cuda/std/__cuda/api_wrapper.h>
#  include <cuda/std/__new/bad_alloc.h>

#  if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR

/**
 * @brief `cuda_pinned_memory_resource` uses cudaMallocHost / cudaFreeHost for allocation/deallocation.
 */
class cuda_pinned_memory_resource
{
private:
  unsigned int __flags_ = cudaHostAllocDefault;

  static constexpr unsigned int __available_flags =
    cudaHostAllocDefault | cudaHostAllocPortable | cudaHostAllocMapped | cudaHostAllocWriteCombined;

public:
  constexpr cuda_pinned_memory_resource(const unsigned int __flags = cudaHostAllocDefault) noexcept
      : __flags_(__flags & __available_flags)
  {
    _LIBCUDACXX_ASSERT(__flags_ == __flags, "Unexpected flags passed to cuda_pinned_memory_resource");
  }

  /**
   * @brief Allocate host memory of size at least \p __bytes.
   * @param __bytes The size in bytes of the allocation.
   * @param __alignment The requested alignment of the allocation.
   * @throw cuda::cuda_error if allocation fails with a CUDA error.
   * @return Pointer to the newly allocated memory
   */
  _CCCL_NODISCARD void* allocate(const size_t __bytes,
                                 const size_t __alignment = default_cuda_malloc_host_alignment) const
  {
    if (!__is_valid_alignment(__alignment))
    {
      _CUDA_VSTD::__throw_bad_alloc();
    }

    void* __ptr{nullptr};
    _CCCL_TRY_CUDA_API(::cudaMallocHost, "Failed to allocate memory with cudaMallocHost.", &__ptr, __bytes, __flags_);
    return __ptr;
  }

  /**
   * @brief Deallocate memory pointed to by \p __ptr.
   * @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate`
   * @param __bytes The number of bytes that was passed to the `allocate` call that returned \p __ptr.
   * @param __alignment The alignment that was passed to the `allocate` call that returned \p __ptr.
   */
  void deallocate(void* __ptr, const size_t, const size_t __alignment = default_cuda_malloc_host_alignment) const
  {
    _LIBCUDACXX_ASSERT(__is_valid_alignment(__alignment),
                       "Invalid alignment passed to cuda_memory_resource::deallocate.");
    _CCCL_ASSERT_CUDA_API(::cudaFreeHost, "cuda_pinned_memory_resource::deallocate failed", __ptr);
    (void) __alignment;
  }

  /**
   * @brief Equality comparison with another cuda_pinned_memory_resource
   * @return Whether both cuda_pinned_memory_resource were constructed with the same flags
   */
  _CCCL_NODISCARD constexpr bool operator==(cuda_pinned_memory_resource const& __other) const noexcept
  {
    return __flags_ == __other.__flags_;
  }
#    if _CCCL_STD_VER <= 2017
  /**
   * @brief Equality comparison with another cuda_pinned_memory_resource
   * @return Whether both cuda_pinned_memory_resource were constructed with different flags
   */
  _CCCL_NODISCARD constexpr bool operator!=(cuda_pinned_memory_resource const& __other) const noexcept
  {
    return __flags_ != __other.__flags_;
  }
#    endif // _CCCL_STD_VER <= 2017

  /**
   * @brief Equality comparison between a cuda_memory_resource and another resource
   * @param __lhs The cuda_memory_resource
   * @param __rhs The resource to compare to
   * @return If the underlying types are equality comparable, returns the result of equality comparison of both
   * resources. Otherwise, returns false.
   */
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator==(cuda_pinned_memory_resource const& __lhs, _Resource const& __rhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_pinned_memory_resource, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_pinned_memory_resource&>(__lhs)}
        == resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
#    if _CCCL_STD_VER <= 2017
  /**
   * @copydoc cuda_pinned_memory_resource::operator<_Resource>==(cuda_pinned_memory_resource const&, _Resource const&)
   */
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator==(_Resource const& __rhs, cuda_pinned_memory_resource const& __lhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_pinned_memory_resource, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_pinned_memory_resource&>(__lhs)}
        == resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
  /**
   * @copydoc cuda_pinned_memory_resource::operator<_Resource>==(cuda_pinned_memory_resource const&, _Resource const&)
   */
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator!=(cuda_pinned_memory_resource const& __lhs, _Resource const& __rhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_pinned_memory_resource, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_pinned_memory_resource&>(__lhs)}
        != resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
  /**
   * @copydoc cuda_pinned_memory_resource::operator<_Resource>==(cuda_pinned_memory_resource const&, _Resource const&)
   */
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator!=(_Resource const& __rhs, cuda_pinned_memory_resource const& __lhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_pinned_memory_resource, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_pinned_memory_resource&>(__lhs)}
        != resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
#    endif // _CCCL_STD_VER <= 2017

  /**
   * @brief Enables the `device_accessible` property
   */
  friend constexpr void get_property(cuda_pinned_memory_resource const&, device_accessible) noexcept {}
  /**
   * @brief Enables the `host_accessible` property
   */
  friend constexpr void get_property(cuda_pinned_memory_resource const&, host_accessible) noexcept {}

  /**
   * @brief Checks whether the passed in alignment is valid
   */
  static constexpr bool __is_valid_alignment(const size_t __alignment) noexcept
  {
    return __alignment <= default_cuda_malloc_host_alignment && (default_cuda_malloc_host_alignment % __alignment == 0);
  }
};
static_assert(resource_with<cuda_pinned_memory_resource, device_accessible>, "");
static_assert(resource_with<cuda_pinned_memory_resource, host_accessible>, "");

_LIBCUDACXX_END_NAMESPACE_CUDA_MR

#  endif // _CCCL_STD_VER >= 2014

#endif // !_CCCL_COMPILER_MSVC_2017 && LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif //_CUDA__MEMORY_RESOURCE_CUDA_PINNED_MEMORY_RESOURCE_H
