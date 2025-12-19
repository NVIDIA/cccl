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

#if _CCCL_HAS_CTK()

#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__type_traits/is_convertible.h>
#  include <cuda/std/__type_traits/is_nothrow_convertible.h>
#  include <cuda/std/__type_traits/is_same.h>

#  include <cuda/std/__cccl/prologue.h>

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
    else if (__i == 1)
    {
      return y;
    }
    else
    {
      return z;
    }
  }
  [[nodiscard]] _CCCL_API constexpr const _Tp& operator[](::cuda::std::size_t __i) const noexcept
  {
    if (__i == 0)
    {
      return x;
    }
    else if (__i == 1)
    {
      return y;
    }
    else
    {
      return z;
    }
  }

  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Tp2, signed char>)
  _CCCL_API constexpr operator char3() const noexcept
  {
    return {static_cast<signed char>(x), static_cast<signed char>(y), static_cast<signed char>(z)};
  }

  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Tp2, short>)
  _CCCL_API constexpr operator short3() const noexcept
  {
    return {static_cast<short>(x), static_cast<short>(y), static_cast<short>(z)};
  }

  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Tp2, int>)
  _CCCL_API constexpr operator int3() const noexcept
  {
    return {static_cast<int>(x), static_cast<int>(y), static_cast<int>(z)};
  }

  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Tp2, long>)
  _CCCL_API constexpr operator long3() const noexcept
  {
    return {static_cast<long>(x), static_cast<long>(y), static_cast<long>(z)};
  }

  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Tp2, long long>)
  _CCCL_API constexpr operator longlong3() const noexcept
  {
    return {static_cast<long long>(x), static_cast<long long>(y), static_cast<long long>(z)};
  }

  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Tp2, unsigned char>)
  _CCCL_API constexpr operator uchar3() const noexcept
  {
    return {static_cast<unsigned char>(x), static_cast<unsigned char>(y), static_cast<unsigned char>(z)};
  }

  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Tp2, unsigned short>)
  _CCCL_API constexpr operator ushort3() const noexcept
  {
    return {static_cast<unsigned short>(x), static_cast<unsigned short>(y), static_cast<unsigned short>(z)};
  }

  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Tp2, unsigned>)
  _CCCL_API constexpr operator uint3() const noexcept
  {
    return {static_cast<unsigned>(x), static_cast<unsigned>(y), static_cast<unsigned>(z)};
  }

  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Tp2, unsigned long>)
  _CCCL_API constexpr operator ulong3() const noexcept
  {
    return {static_cast<unsigned long>(x), static_cast<unsigned long>(y), static_cast<unsigned long>(z)};
  }

  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Tp2, unsigned long long>)
  _CCCL_API constexpr operator ulonglong3() const noexcept
  {
    return {static_cast<unsigned long long>(x), static_cast<unsigned long long>(y), static_cast<unsigned long long>(z)};
  }
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_HIERARCHY_QUERY_RESULT_H
