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
#include <cuda/__utility/__basic_any/semiregular.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

template <class _Resource, class _OtherResource>
_CCCL_CONCEPT __non_polymorphic_resources = _CCCL_REQUIRES_EXPR((_Resource, _OtherResource))(
  requires(::cuda::mr::synchronous_resource<_Resource>),
  requires(::cuda::mr::synchronous_resource<_OtherResource>),
  requires(__non_polymorphic<_Resource>),
  requires(__non_polymorphic<_OtherResource>));

//! @brief Equality comparison between two resources of different types. Always returns false.
_CCCL_TEMPLATE(class _Resource, class _OtherResource)
_CCCL_REQUIRES(
  (!::cuda::std::is_same_v<_Resource, _OtherResource>) _CCCL_AND __non_polymorphic_resources<_Resource, _OtherResource>)
[[nodiscard]] bool operator==(_Resource const&, _OtherResource const&) noexcept
{
  return false;
}

#if _CCCL_STD_VER <= 2017
//! @brief Inequality comparison between two resources of different types. Always returns true.
_CCCL_TEMPLATE(class _Resource, class _OtherResource)
_CCCL_REQUIRES(
  (!::cuda::std::is_same_v<_Resource, _OtherResource>) _CCCL_AND __non_polymorphic_resources<_Resource, _OtherResource>)
[[nodiscard]] bool operator!=(_Resource const&, _OtherResource const&) noexcept
{
  return true;
}
#endif // _CCCL_STD_VER <= 2017

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //_CUDAX__MEMORY_RESOURCE_RESOURCE_CUH
