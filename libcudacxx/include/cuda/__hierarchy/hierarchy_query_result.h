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

// #include <cuda/__type_traits/vector_type.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_nothrow_convertible.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp, ::cuda::std::size_t _Np>
struct hierarchy_query_result
{
  using value_type                            = _Tp;
  static constexpr ::cuda::std::size_t __rank = _Np;

  _Tp __rest_[_Np - 3];
  _Tp z;
  _Tp y;
  _Tp x;

  _CCCL_HIDE_FROM_ABI constexpr hierarchy_query_result() noexcept = default;

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((sizeof...(_Args) == _Np) _CCCL_AND(::cuda::std::is_convertible_v<_Args, _Tp>&&...))
  _CCCL_API constexpr hierarchy_query_result(_Args&&... __args) noexcept(
    (::cuda::std::is_nothrow_convertible_v<_Args, _Tp> && ...))
      : __rest_{}
      , z{}
      , y{}
      , x{}
  {
    _Tp __tmp[]{__args...};
    for (::cuda::std::size_t __i = 0; __i < _Np; ++__i)
    {
      operator[](__i) = __tmp[__i];
    }
  }

  _CCCL_HIDE_FROM_ABI constexpr hierarchy_query_result(const hierarchy_query_result&) noexcept = default;
  _CCCL_HIDE_FROM_ABI constexpr hierarchy_query_result(hierarchy_query_result&&) noexcept      = default;

  [[nodiscard]] _CCCL_API constexpr _Tp& operator[](::cuda::std::size_t __i) noexcept
  {
    if (__i < _Np - 3)
    {
      return __rest_[__i];
    }
    if (__i == _Np - 3)
    {
      return z;
    }
    if (__i == _Np - 2)
    {
      return y;
    }
    if (__i == _Np - 1)
    {
      return x;
    }
    _CCCL_UNREACHABLE();
  }
  [[nodiscard]] _CCCL_API constexpr const _Tp& operator[](::cuda::std::size_t __i) const noexcept
  {
    if (__i < _Np - 3)
    {
      return __rest_[__i];
    }
    if (__i == _Np - 3)
    {
      return z;
    }
    if (__i == _Np - 2)
    {
      return y;
    }
    if (__i == _Np - 1)
    {
      return x;
    }
    _CCCL_UNREACHABLE();
  }

  // _CCCL_TEMPLATE(class _Tp2 = _Tp, ::cuda::std::size_t _Np2 = _Np)
  // _CCCL_REQUIRES(__has_vector_type_v<_Tp, _Np2>)
  // _CCCL_API constexpr operator __vector_type_t<_Tp2, _Np2>() const noexcept
  // {
  //   // only 4 element vector types will be instantiated
  //   __vector_type_t<_Tp2, _Np2> __ret{};
  //   __ret.x = x;
  //   __ret.y = y;
  //   __ret.z = z;
  //   __ret.w = operator[](_Np2 - 4);
  //   return __ret;
  // }
};

template <class _Tp>
struct hierarchy_query_result<_Tp, 0>
{
  using value_type                            = _Tp;
  static constexpr ::cuda::std::size_t __rank = 0;

  [[nodiscard]] _CCCL_API constexpr _Tp& operator[](::cuda::std::size_t) noexcept
  {
    _CCCL_UNREACHABLE();
  }
  [[nodiscard]] _CCCL_API constexpr const _Tp& operator[](::cuda::std::size_t) const noexcept
  {
    _CCCL_UNREACHABLE();
  }
};

template <class _Tp>
struct hierarchy_query_result<_Tp, 1>
{
  using value_type                            = _Tp;
  static constexpr ::cuda::std::size_t __rank = 1;

  _Tp x;

  [[nodiscard]] _CCCL_API constexpr _Tp& operator[](::cuda::std::size_t __i) noexcept
  {
    if (__i == 0)
    {
      return x;
    }
    _CCCL_UNREACHABLE();
  }
  [[nodiscard]] _CCCL_API constexpr const _Tp& operator[](::cuda::std::size_t __i) const noexcept
  {
    if (__i == 0)
    {
      return x;
    }
    _CCCL_UNREACHABLE();
  }

  // _CCCL_TEMPLATE(class _Tp2 = _Tp)
  // _CCCL_REQUIRES(__has_vector_type_v<_Tp, 1>)
  // _CCCL_API constexpr operator __vector_type_t<_Tp2, 1>() const noexcept
  // {
  //   __vector_type_t<_Tp2, 1> __ret{};
  //   __ret.x = x;
  //   return __ret;
  // }
};

template <class _Tp>
struct hierarchy_query_result<_Tp, 2>
{
  using value_type                            = _Tp;
  static constexpr ::cuda::std::size_t __rank = 2;

  _Tp y;
  _Tp x;

  [[nodiscard]] _CCCL_API constexpr _Tp& operator[](::cuda::std::size_t __i) noexcept
  {
    if (__i == 0)
    {
      return y;
    }
    if (__i == 1)
    {
      return x;
    }
    _CCCL_UNREACHABLE();
  }
  [[nodiscard]] _CCCL_API constexpr const _Tp& operator[](::cuda::std::size_t __i) const noexcept
  {
    if (__i == 0)
    {
      return y;
    }
    if (__i == 1)
    {
      return x;
    }
    _CCCL_UNREACHABLE();
  }

  // _CCCL_TEMPLATE(class _Tp2 = _Tp, )
  // _CCCL_REQUIRES(__has_vector_type_v<_Tp, 2>)
  // _CCCL_API constexpr operator __vector_type_t<_Tp2, 2>() const noexcept
  // {
  //   __vector_type_t<_Tp2, 2> __ret{};
  //   __ret.x = x;
  //   __ret.y = y;
  //   return __ret;
  // }
};

template <class _Tp>
struct hierarchy_query_result<_Tp, 3>
{
  using value_type                            = _Tp;
  static constexpr ::cuda::std::size_t __rank = 3;

  _Tp z;
  _Tp y;
  _Tp x;

  [[nodiscard]] _CCCL_API constexpr _Tp& operator[](::cuda::std::size_t __i) noexcept
  {
    if (__i == 0)
    {
      return z;
    }
    if (__i == 1)
    {
      return y;
    }
    if (__i == 2)
    {
      return x;
    }
    _CCCL_UNREACHABLE();
  }
  [[nodiscard]] _CCCL_API constexpr const _Tp& operator[](::cuda::std::size_t __i) const noexcept
  {
    if (__i == 0)
    {
      return z;
    }
    if (__i == 1)
    {
      return y;
    }
    if (__i == 2)
    {
      return x;
    }
    _CCCL_UNREACHABLE();
  }

  // _CCCL_TEMPLATE(class _Tp2 = _Tp)
  // _CCCL_REQUIRES(__has_vector_type_v<_Tp, 3>)
  // _CCCL_API constexpr operator __vector_type_t<_Tp2, 3>() const noexcept
  // {
  //   __vector_type_t<_Tp2, 3> __ret{};
  //   __ret.x = x;
  //   __ret.y = y;
  //   __ret.z = z;
  //   return __ret;
  // }

  // _CCCL_TEMPLATE(class _Tp2 = _Tp)
  // _CCCL_REQUIRES(::cuda::std::is_same_v<_Tp2, unsigned>)
  // _CCCL_API constexpr operator ::dim3() const noexcept
  // {
  //   return ::dim3{operator uint3()};
  // }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___HIERARCHY_HIERARCHY_QUERY_RESULT_H
