//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

namespace cuda::experimental
{

#if _CCCL_STD_VER >= 2020
//! @brief Equality comparison between two resources.
//! @param __lhs The left-hand side resource.
//! @param __rhs The right-hand side resource.
//! @returns If the underlying types are equality comparable, returns the result of equality comparison of both
//! resources. Otherwise, returns false.
template <class _Resource, class _OtherResource>
  requires _CUDA_VMR::resource<_Resource> && _CUDA_VMR::resource<_OtherResource>
        && _CUDA_VMR::__different_resource<_Resource, _OtherResource> && __non_polymorphic<_Resource>
        && __non_polymorphic<_OtherResource>
_CCCL_NODISCARD bool
operator==([[maybe_unused]] _Resource const& __lhs, [[maybe_unused]] _OtherResource const& __rhs) noexcept
{
  return false;
}

#else // ^^^ C++20 ^^^ / vvv C++17
template <class _Resource, class _OtherResource>
_CCCL_NODISCARD auto
operator==([[maybe_unused]] _Resource const& __lhs, [[maybe_unused]] _OtherResource const& __rhs) noexcept
  _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VMR::resource<_Resource>&& _CUDA_VMR::resource<_OtherResource>&&
                                  _CUDA_VMR::__different_resource<_Resource, _OtherResource>&&
                                    __non_polymorphic<_Resource>&& __non_polymorphic<_OtherResource>)
{
  return false;
}

template <class _Resource, class _OtherResource>
_CCCL_NODISCARD auto
operator!=([[maybe_unused]] _Resource const& __lhs, [[maybe_unused]] _OtherResource const& __rhs) noexcept
  _CCCL_TRAILING_REQUIRES(bool)(_CUDA_VMR::resource<_Resource>&& _CUDA_VMR::resource<_OtherResource>&&
                                  _CUDA_VMR::__different_resource<_Resource, _OtherResource>&&
                                    __non_polymorphic<_Resource>&& __non_polymorphic<_OtherResource>)
{
  return true;
}

#endif // _CCCL_STD_VER <= 2017

} // namespace cuda::experimental

#endif //_CUDAX__MEMORY_RESOURCE_RESOURCE_CUH
