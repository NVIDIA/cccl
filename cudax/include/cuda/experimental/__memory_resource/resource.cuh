//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_RESOURCE_CUH
#define _CUDAX__MEMORY_RESOURCE_RESOURCE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory_resource/resource.h>

#include <cuda/experimental/__utility/basic_any/semiregular.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

// Outline to a struct, because resource concept relies on equality comparison, so we can run into a circular dependency
// when below operator== is taken into consideration for equality comparison trait.
template <bool _IsDifferent>
struct __different_non_polymorphic_resources_impl
{
  template <class _Resource, class _OtherResource>
  static constexpr bool __value(_Resource*, _OtherResource*) noexcept
  {
    return false;
  }
};

template <>
struct __different_non_polymorphic_resources_impl<true>
{
  template <class _Resource, class _OtherResource>
  static constexpr bool __value(_Resource*, _OtherResource*) noexcept
  {
    return _CUDA_VMR::resource<_Resource> && _CUDA_VMR::resource<_OtherResource> && __non_polymorphic<_Resource>
        && __non_polymorphic<_OtherResource>;
  }
};

template <class _Resource, class _OtherResource>
_CCCL_CONCEPT __different_non_polymorphic_resources =
  __different_non_polymorphic_resources_impl<_CUDA_VMR::__different_resource<_Resource, _OtherResource>>::__value(
    static_cast<_Resource*>(nullptr), static_cast<_OtherResource*>(nullptr));

//! @brief Equality comparison between two resources of different types.
//! @param __lhs The left-hand side resource.
//! @param __rhs The right-hand side resource.
//! @returns Always returns false.
_CCCL_TEMPLATE(class _Resource, class _OtherResource)
_CCCL_REQUIRES(__different_non_polymorphic_resources<_Resource, _OtherResource>)
[[nodiscard]] bool operator==(_Resource const&, _OtherResource const&) noexcept
{
  return false;
}

#if _CCCL_STD_VER <= 2017
//! @brief Inequality comparison between two resources of different types.
//! @param __lhs The left-hand side resource.
//! @param __rhs The right-hand side resource.
//! @returns Always returns true.
_CCCL_TEMPLATE(class _Resource, class _OtherResource)
_CCCL_REQUIRES(__different_non_polymorphic_resources<_Resource, _OtherResource>)
[[nodiscard]] bool operator!=(_Resource const&, _OtherResource const&) noexcept
{
  return true;
}
#endif // _CCCL_STD_VER <= 2017

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //_CUDAX__MEMORY_RESOURCE_RESOURCE_CUH
