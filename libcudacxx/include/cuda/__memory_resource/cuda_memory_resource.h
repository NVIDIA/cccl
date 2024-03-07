//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MEMORY_RESOURCE_CUDA_MEMORY_RESOURCE_H
#define _CUDA__MEMORY_RESOURCE_CUDA_MEMORY_RESOURCE_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !defined(_CCCL_CUDA_COMPILER_NVCC) && !defined(_CCCL_CUDA_COMPILER_NVHPC)
#  include <cuda_runtime_api.h>
#endif // !_CCCL_CUDA_COMPILER_NVCC && !_CCCL_CUDA_COMPILER_NVHPC

#include <cuda/__memory_resource/get_property.h>
#include <cuda/__memory_resource/properties.h>
#include <cuda/__memory_resource/resource_ref.h>
#include <cuda/__memory_resource/resource.h>

#if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR

/**
 * @brief `cuda_memory_resource` uses cudaMalloc / cudaFree for allocation/deallocation.
 */
struct cuda_memory_resource
{
  /**
   * @brief Allocate device memory of size at least \p __bytes.
   * @param __bytes The size in bytes of the allocation.
   * @param __alignment The requested alignment of the allocation. Is ignored!
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate(const size_t __bytes, const size_t __alignment) const
  {
    _LIBCUDACXX_ASSERT(__alignment <= 256 && (256 % __alignment == 0),
                       "cuda_memory_resource::allocate invalid alignment");
    return allocate(__bytes);
  }

  /**
   * @brief Allocate device memory of size at least \p __bytes.
   * @param __bytes The size in bytes of the allocation.
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate(const size_t __bytes) const
  {
    void* __ptr{nullptr};
    const ::cudaError_t __status = ::cudaMalloc(&__ptr, __bytes);
    switch (__status)
    {
      case ::cudaSuccess:
        return __ptr;
      default:
        ::cudaGetLastError(); // Clear CUDA error state
#  ifndef _LIBCUDACXX_NO_EXCEPTIONS
        throw cuda::cuda_error{__status, "Failed to allocate memory with cudaMalloc."};
#  else
        _LIBCUDACXX_UNREACHABLE();
#  endif
    }
  }

  /**
   * @brief Deallocate memory pointed to by \p __ptr.
   * @param __ptr Pointer to be deallocated
   * @param __bytes The size in bytes of the allocation.
   * @param __alignment The alignment that was passed to the `allocate` call that returned \p __ptr. Is ignored!
   */
  void deallocate(void* __ptr, const size_t __bytes, const size_t __alignment) const
  {
    _LIBCUDACXX_ASSERT(__alignment <= 256 && (256 % __alignment == 0),
                       "cuda_memory_resource::deallocate invalid alignment");
    deallocate(__ptr, __bytes);
  }

  /**
   * @brief Deallocate memory pointed to by \p __ptr.
   * @param __ptr Pointer to be deallocated
   * @param __bytes The size in bytes of the allocation.
   */
  void deallocate(void* __ptr, size_t) const
  {
    const ::cudaError_t __status = ::cudaFree(__ptr);
    (void)__status;
    _LIBCUDACXX_ASSERT(__status == cudaSuccess, "cuda_memory_resource::deallocate failed");
  }

  /**
   * @brief Equality comparison operator between two cuda_memory_resource's
   * @return true
   */
  _LIBCUDACXX_NODISCARD_ATTRIBUTE constexpr bool operator==(cuda_memory_resource const&) const noexcept
  {
    return true;
  }
#  if _CCCL_STD_VER <= 2017
  /**
   * @brief Inequality comparison operator between two cuda_memory_resource's
   * @return false
   */
  _LIBCUDACXX_NODISCARD_ATTRIBUTE constexpr bool operator!=(cuda_memory_resource const&) const noexcept
  {
    return false;
  }
#  endif // _CCCL_STD_VER <= 2017

  /**
   * @brief Equality comparison operator between a cuda_memory_resource and a device_accessible resource
   *
   * @param __lhs The cuda_memory_resource
   * @param __rhs The resource to compare to
   * @return false
   */
  _LIBCUDACXX_TEMPLATE(class _Resource)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_Resource, cuda_memory_resource>)
                         _LIBCUDACXX_AND resource<_Resource> _LIBCUDACXX_AND has_property<_Resource, device_accessible>)
  _LIBCUDACXX_NODISCARD_FRIEND bool operator==(cuda_memory_resource const& __lhs, _Resource const& __rhs) noexcept
  {
    return resource_ref<device_accessible>{const_cast<cuda_memory_resource&>(__lhs)}
        == resource_ref<device_accessible>{const_cast<_Resource&>(__rhs)};
  }
#  if _CCCL_STD_VER <= 2017
  _LIBCUDACXX_TEMPLATE(class _Resource)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_Resource, cuda_memory_resource>)
                         _LIBCUDACXX_AND resource<_Resource> _LIBCUDACXX_AND has_property<_Resource, device_accessible>)
  _LIBCUDACXX_NODISCARD_FRIEND bool operator==(_Resource const& __rhs, cuda_memory_resource const& __lhs) noexcept
  {
    return resource_ref<device_accessible>{const_cast<cuda_memory_resource&>(__lhs)}
        == resource_ref<device_accessible>{const_cast<_Resource&>(__rhs)};
  }
  _LIBCUDACXX_TEMPLATE(class _Resource)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_Resource, cuda_memory_resource>)
                         _LIBCUDACXX_AND resource<_Resource> _LIBCUDACXX_AND has_property<_Resource, device_accessible>)
  _LIBCUDACXX_NODISCARD_FRIEND bool operator!=(cuda_memory_resource const& __lhs, _Resource const& __rhs) noexcept
  {
    return resource_ref<device_accessible>{const_cast<cuda_memory_resource&>(__lhs)}
        != resource_ref<device_accessible>{const_cast<_Resource&>(__rhs)};
  }
  _LIBCUDACXX_TEMPLATE(class _Resource)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_Resource, cuda_memory_resource>)
                         _LIBCUDACXX_AND resource<_Resource> _LIBCUDACXX_AND has_property<_Resource, device_accessible>)
  _LIBCUDACXX_NODISCARD_FRIEND bool operator!=(_Resource const& __rhs, cuda_memory_resource const& __lhs) noexcept
  {
    return resource_ref<device_accessible>{const_cast<cuda_memory_resource&>(__lhs)}
        != resource_ref<device_accessible>{const_cast<_Resource&>(__rhs)};
  }
#  endif // _CCCL_STD_VER <= 2017

  /**
   * @brief Equality comparison operator between a cuda_memory_resource and an arbitrary resource
   *
   * @param __lhs The cuda_memory_resource
   * @param __rhs The resource to compare to
   * @return false
   */
  _LIBCUDACXX_TEMPLATE(class _Resource)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_Resource, cuda_memory_resource>) _LIBCUDACXX_AND resource<_Resource>
                         _LIBCUDACXX_AND(!has_property<_Resource, device_accessible>))
  _LIBCUDACXX_NODISCARD_FRIEND bool operator==(cuda_memory_resource const&, _Resource const&) noexcept
  {
    return false;
  }
#  if _CCCL_STD_VER <= 2017
  _LIBCUDACXX_TEMPLATE(class _Resource)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_Resource, cuda_memory_resource>) _LIBCUDACXX_AND resource<_Resource>
                         _LIBCUDACXX_AND(!has_property<_Resource, device_accessible>))
  _LIBCUDACXX_NODISCARD_FRIEND bool operator==(_Resource const&, cuda_memory_resource const&) noexcept
  {
    return false;
  }
  _LIBCUDACXX_TEMPLATE(class _Resource)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_Resource, cuda_memory_resource>) _LIBCUDACXX_AND resource<_Resource>
                         _LIBCUDACXX_AND(!has_property<_Resource, device_accessible>))
  _LIBCUDACXX_NODISCARD_FRIEND bool operator!=(cuda_memory_resource const&, _Resource const&) noexcept
  {
    return true;
  }
  _LIBCUDACXX_TEMPLATE(class _Resource)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_Resource, cuda_memory_resource>) _LIBCUDACXX_AND resource<_Resource>
                         _LIBCUDACXX_AND(!has_property<_Resource, device_accessible>))
  _LIBCUDACXX_NODISCARD_FRIEND bool operator!=(_Resource const&, cuda_memory_resource const&) noexcept
  {
    return true;
  }
#  endif // _CCCL_STD_VER <= 2017

  /**
   * @brief Enables the `device_accessible` property
   */
  friend constexpr void get_property(cuda_memory_resource const&, device_accessible) noexcept {}
};
static_assert(resource_with<cuda_memory_resource, device_accessible>, "");

_LIBCUDACXX_END_NAMESPACE_CUDA_MR

#endif // _CCCL_STD_VER >= 2014

#endif //_CUDA__MEMORY_RESOURCE_CUDA_MEMORY_RESOURCE_H
