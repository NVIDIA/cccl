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
#  include <cuda/std/__concepts/__concept_macros.h>
#  include <cuda/std/__cuda/api_wrapper.h>
#  include <cuda/std/__cuda/ensure_current_device.h>
#  include <cuda/std/detail/libcxx/include/stdexcept>

#  if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR

//! @brief device_memory_resource uses `cudaMalloc` / `cudaFree` for allocation / deallocation.
//! By default uses device 0 to allocate memory
class device_memory_resource
{
private:
  int __device_id_{0};

public:
  //! @brief default constructs a device_memory_resource allocating memory on device 0
  _CCCL_HIDE_FROM_ABI device_memory_resource() = default;

  //! @brief default constructs a device_memory_resource allocating memory on device \p __device_id
  //! @param __device_id The id of the device we are allocating memory on
  constexpr device_memory_resource(const int __device_id) noexcept
      : __device_id_(__device_id)
  {}

  //! @brief Allocate device memory of size at least \p __bytes.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @throw std::invalid_argument in case of invalid alignment or \c cuda::cuda_error of the returned error code.
  //! @return Pointer to the newly allocated memory
  _CCCL_NODISCARD void* allocate(const size_t __bytes, const size_t __alignment = default_cuda_malloc_alignment) const
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    if (!__is_valid_alignment(__alignment))
    {
      _CUDA_VSTD::__throw_invalid_argument("Invalid alignment passed to device_memory_resource::allocate.");
    }

    // We need to ensure that we allocate on the right device as `cudaMalloc` always uses the current device
    __ensure_current_device __device_wrapper{__device_id_};

    void* __ptr{nullptr};
    _CCCL_TRY_CUDA_API(::cudaMalloc, "Failed to allocate memory with cudaMalloc.", &__ptr, __bytes);
    return __ptr;
  }

  //! @brief Deallocate memory pointed to by \p __ptr.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a call to `allocate`
  //! @param __bytes The number of bytes that was passed to the `allocate` call that returned \p __ptr.
  //! @param __alignment The alignment that was passed to the `allocate` call that returned \p __ptr.
  void deallocate(void* __ptr, const size_t, const size_t __alignment = default_cuda_malloc_alignment) const noexcept
  {
    // We need to ensure that the provided alignment matches the minimal provided alignment
    _CCCL_ASSERT(__is_valid_alignment(__alignment), "Invalid alignment passed to device_memory_resource::deallocate.");
    _CCCL_ASSERT_CUDA_API(::cudaFree, "device_memory_resource::deallocate failed", __ptr);
    (void) __alignment;
  }

  //! @brief Equality comparison with another \c device_memory_resource
  //! @param __other The other \c device_memory_resource
  //! @return true, if both resources hold the same device id
  _CCCL_NODISCARD constexpr bool operator==(device_memory_resource const& __other) const noexcept
  {
    return __device_id_ == __other.__device_id_;
  }
#    if _CCCL_STD_VER <= 2017
  //! @brief Inequality comparison with another \c device_memory_resource
  //! @param __other The other \c device_memory_resource
  //! @return true, if both resources hold different device id's
  _CCCL_NODISCARD constexpr bool operator!=(device_memory_resource const& __other) const noexcept
  {
    return __device_id_ != __other.__device_id_;
  }
#    endif // _CCCL_STD_VER <= 2017

#    if _CCCL_STD_VER >= 2020
  //! @brief Equality comparison between a \c device_memory_resource and another resource
  //! @param __rhs The resource to compare to
  //! @return If the underlying types are equality comparable, returns the result of equality comparison of both
  //! resources. Otherwise, returns false.
  _LIBCUDACXX_TEMPLATE(class _Resource)
  _LIBCUDACXX_REQUIRES((__different_resource<device_memory_resource, _Resource>) )
  _CCCL_NODISCARD bool operator==(_Resource const& __rhs) const noexcept
  {
    if constexpr (has_property<_Resource, device_accessible>)
    {
      return resource_ref<device_accessible>{const_cast<device_memory_resource*>(this)}
          == resource_ref<device_accessible>{const_cast<_Resource&>(__rhs)};
    }
    else
    {
      return false;
    }
  }
#    else // ^^^ C++20 ^^^ / vvv C++17
  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator==(device_memory_resource const& __lhs, _Resource const& __rhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(
      __different_resource<device_memory_resource, _Resource>&& has_property<_Resource, device_accessible>)
  {
    return resource_ref<device_accessible>{const_cast<device_memory_resource&>(__lhs)}
        == resource_ref<device_accessible>{const_cast<_Resource&>(__rhs)};
  }

  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator==(device_memory_resource const&, _Resource const&) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<device_memory_resource, _Resource>
                                        && !has_property<_Resource, device_accessible>)
  {
    return false;
  }

  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator==(_Resource const& __rhs, device_memory_resource const& __lhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(
      __different_resource<device_memory_resource, _Resource>&& has_property<_Resource, device_accessible>)
  {
    return resource_ref<device_accessible>{const_cast<device_memory_resource&>(__lhs)}
        == resource_ref<device_accessible>{const_cast<_Resource&>(__rhs)};
  }

  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator==(_Resource const&, device_memory_resource const&) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<device_memory_resource, _Resource>
                                        && !has_property<_Resource, device_accessible>)
  {
    return false;
  }

  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator!=(device_memory_resource const& __lhs, _Resource const& __rhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(
      __different_resource<device_memory_resource, _Resource>&& has_property<_Resource, device_accessible>)
  {
    return resource_ref<device_accessible>{const_cast<device_memory_resource&>(__lhs)}
        != resource_ref<device_accessible>{const_cast<_Resource&>(__rhs)};
  }

  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator!=(device_memory_resource const&, _Resource const&) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<device_memory_resource, _Resource>
                                        && !has_property<_Resource, device_accessible>)
  {
    return true;
  }

  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator!=(_Resource const& __rhs, device_memory_resource const& __lhs) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(
      __different_resource<device_memory_resource, _Resource>&& has_property<_Resource, device_accessible>)
  {
    return resource_ref<device_accessible>{const_cast<device_memory_resource&>(__lhs)}
        != resource_ref<device_accessible>{const_cast<_Resource&>(__rhs)};
  }

  template <class _Resource>
  _CCCL_NODISCARD_FRIEND auto operator!=(_Resource const&, device_memory_resource const&) noexcept
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__different_resource<device_memory_resource, _Resource>
                                        && !has_property<_Resource, device_accessible>)
  {
    return true;
  }
#    endif // _CCCL_STD_VER <= 2017

  //! @brief Enables the \c device_accessible property
  friend constexpr void get_property(device_memory_resource const&, device_accessible) noexcept {}

  //! @brief Checks whether the passed in alignment is valid
  static constexpr bool __is_valid_alignment(const size_t __alignment) noexcept
  {
    return __alignment <= default_cuda_malloc_alignment && (default_cuda_malloc_alignment % __alignment == 0);
  }
};
static_assert(resource_with<device_memory_resource, device_accessible>, "");

// For backward compatability
using cuda_memory_resource _LIBCUDACXX_DEPRECATED = device_memory_resource;

_LIBCUDACXX_END_NAMESPACE_CUDA_MR

#  endif // _CCCL_STD_VER >= 2014

#endif // !_CCCL_COMPILER_MSVC_2017 && LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif // _CUDA__MEMORY_RESOURCE_CUDA_MEMORY_RESOURCE_H
