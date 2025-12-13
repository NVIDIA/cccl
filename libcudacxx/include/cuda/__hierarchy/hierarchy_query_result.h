//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_HIERARCHY_QUERY_RESULT_H
#define _CUDA___HIERARCHY_HIERARCHY_QUERY_RESULT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/vector_type.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_nothrow_convertible.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp>
struct hierarchy_query_result
{
  using value_type = _Tp;

  _Tp x;
  _Tp y;
  _Tp z;

  [[nodiscard]] _CCCL_API constexpr _Tp& operator[](::cuda::std::size_t __i) noexcept
  {
    if (__i == 0)
    {
      return x;
    }
    if (__i == 1)
    {
      return y;
    }
    if (__i == 2)
    {
      return z;
    }
    _CCCL_UNREACHABLE();
  }
  [[nodiscard]] _CCCL_API constexpr const _Tp& operator[](::cuda::std::size_t __i) const noexcept
  {
    if (__i == 0)
    {
      return x;
    }
    if (__i == 1)
    {
      return y;
    }
    if (__i == 2)
    {
      return z;
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(__has_vector_type_v<_Tp2, 3>)
  _CCCL_API constexpr operator __vector_type_t<_Tp2, 3>() const noexcept
  {
    __vector_type_t<_Tp, 3> __ret{};
    __ret.x = x;
    __ret.y = y;
    __ret.z = z;
    return __ret;
  }

  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Tp2, unsigned>)
  _CCCL_API constexpr operator ::dim3() const noexcept
  {
    return ::dim3{static_cast<::uint3>(*this)};
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___HIERARCHY_HIERARCHY_QUERY_RESULT_H
