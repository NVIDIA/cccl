//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MEMORY_RESOURCE_CUDA_MANAGED_MEMORY_RESOURCE_H
#define _CUDA__MEMORY_RESOURCE_CUDA_MANAGED_MEMORY_RESOURCE_H

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
#    include <cuda_runtime_api.h>
#  endif // !_CCCL_CUDA_COMPILER_NVCC && !_CCCL_CUDA_COMPILER_NVHPC

#  include <cuda/__memory_resource/get_property.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__memory_resource/resource.h>
#  include <cuda/__memory_resource/resource_ref.h>
#  include <cuda/std/__cuda/api_wrapper.h>
#  include <cuda/std/__new/bad_alloc.h>

#  if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR

//! @brief \c cuda_managed_memory_resource uses `cudaMallocManaged` / `cudaFree` for allocation / deallocation.
class cuda_managed_memory_resource
{
private:
  unsigned int __flags_ = cudaMemAttachGlobal;

  static constexpr unsigned int __available_flags = cudaMemAttachGlobal | cudaMemAttachHost;

public:
  constexpr cuda_managed_memory_resource(const unsigned int __flags = cudaMemAttachGlobal) noexcept
      : __flags_(__flags & __available_flags)
  {
    _LIBCUDACXX_ASSERT(__flags_ == __flags, "Unexpected flags passed to cuda_managed_memory_resource");
  }

  //! @brief Allocate CUDA unified memory of size at least \p __bytes.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @throw cuda::cuda_error of the returned error code
  //! @return Pointer to the newly allocated memory
  _CCCL_NODISCARD void* allocate(const size_t __bytes, const size_t __alignment = default_cuda_malloc_alignment) const
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    if (!__is_valid_alignment(__alignment))
    {
      _CUDA_VSTD::__throw_bad_alloc();
    }

    void* __ptr{nullptr};
    _CCCL_TRY_CUDA_API(
      ::cudaMallocManaged, "Failed to allocate memory with cudaMallocManaged.", &__ptr, __bytes, __flags_);
    return __ptr;
  }

  //! @brief Deallocate memory pointed to by \p __ptr.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate`
  //! @param __bytes The number of bytes that was passed to the `allocate` call that returned \p __ptr.
  //! @param __alignment The alignment that was passed to the `allocate` call that returned \p __ptr.
  void deallocate(void* __ptr, const size_t, const size_t __alignment = default_cuda_malloc_alignment) const
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    _LIBCUDACXX_ASSERT(__is_valid_alignment(__alignment),
                       "Invalid alignment passed to cuda_memory_resource::deallocate.");
    _CCCL_ASSERT_CUDA_API(::cudaFree, "cuda_managed_memory_resource::deallocate failed", __ptr);
    (void) __alignment;
  }

  //! @brief Equality comparison with another \c cuda_managed_memory_resource
  //! @param __other The other \c cuda_managed_memory_resource
  //! @return Whether both \c cuda_managed_memory_resource were constructed with the same flags
  _CCCL_NODISCARD constexpr bool operator==(cuda_managed_memory_resource const& __other) const noexcept
  {
    return __flags_ == __other.__flags_;
  }
#    if _CCCL_STD_VER <= 2017
  //! @brief Inequality comparison with another \c cuda_managed_memory_resource
  //! @param __other The other \c cuda_managed_memory_resource
  //! @return Whether both \c cuda_managed_memory_resource were constructed with different flags
  _CCCL_NODISCARD constexpr bool operator!=(cuda_managed_memory_resource const& __other) const noexcept
  {
    return __flags_ != __other.__flags_;
  }
#    endif // _CCCL_STD_VER <= 2017

  //! @brief Equality comparison between a \c cuda_memory_resource and another resource
  //! @param __lhs The \c cuda_memory_resource
  //! @param __rhs The resource to compare to
  //! @return If the underlying types are equality comparable, returns the result of equality comparison of both
  //! resources. Otherwise, returns false.
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator==(cuda_managed_memory_resource const& __lhs, _Resource const& __rhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_managed_memory_resource, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_managed_memory_resource&>(__lhs)}
        == resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
#    if _CCCL_STD_VER <= 2017
  //! @copydoc cuda_managed_memory_resource::operator<_Resource>==(cuda_managed_memory_resource const&, _Resource
  //! const&)
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator==(_Resource const& __rhs, cuda_managed_memory_resource const& __lhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_managed_memory_resource, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_managed_memory_resource&>(__lhs)}
        == resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
  //! @copydoc cuda_managed_memory_resource::operator<_Resource>==(cuda_managed_memory_resource const&, _Resource
  //! const&)
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator!=(cuda_managed_memory_resource const& __lhs, _Resource const& __rhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_managed_memory_resource, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_managed_memory_resource&>(__lhs)}
        != resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
  //! @copydoc cuda_managed_memory_resource::operator<_Resource>==(cuda_managed_memory_resource const&, _Resource
  //! const&)
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator!=(_Resource const& __rhs, cuda_managed_memory_resource const& __lhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<cuda_managed_memory_resource, _Resource>)
  {
    return resource_ref<>{const_cast<cuda_managed_memory_resource&>(__lhs)}
        != resource_ref<>{const_cast<_Resource&>(__rhs)};
  }
#    endif // _CCCL_STD_VER <= 2017

  //! @brief Enables the \c device_accessible property
  friend constexpr void get_property(cuda_managed_memory_resource const&, device_accessible) noexcept {}
  //! @brief Enables the \c host_accessible property
  friend constexpr void get_property(cuda_managed_memory_resource const&, host_accessible) noexcept {}

  //! @brief Checks whether the passed in alignment is valid
  static constexpr bool __is_valid_alignment(const size_t __alignment) noexcept
  {
    return __alignment <= default_cuda_malloc_alignment && (default_cuda_malloc_alignment % __alignment == 0);
  }
};
static_assert(resource_with<cuda_managed_memory_resource, device_accessible>, "");
static_assert(resource_with<cuda_managed_memory_resource, host_accessible>, "");

_LIBCUDACXX_END_NAMESPACE_CUDA_MR

#  endif // _CCCL_STD_VER >= 2014

#endif // !_CCCL_COMPILER_MSVC_2017 && LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif //_CUDA__MEMORY_RESOURCE_CUDA_MANAGED_MEMORY_RESOURCE_H
