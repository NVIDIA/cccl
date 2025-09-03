// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___NUMERIC_MIDPOINT_H
#define _CUDA_STD___NUMERIC_MIDPOINT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_null_pointer.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/cstddef>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
[[nodiscard]]
_CCCL_API constexpr enable_if_t<is_integral_v<_Tp> && !is_same_v<bool, _Tp> && !is_null_pointer_v<_Tp>, _Tp>
midpoint(_Tp __a, _Tp __b) noexcept
{
  using _Up = make_unsigned_t<_Tp>;

  if (__a > __b)
  {
    const _Up __diff = _Up(__a) - _Up(__b);
    return static_cast<_Tp>(__a - static_cast<_Tp>(__diff / 2));
  }
  else
  {
    const _Up __diff = _Up(__b) - _Up(__a);
    return static_cast<_Tp>(__a + static_cast<_Tp>(__diff / 2));
  }
}

template <class _Tp, enable_if_t<is_object_v<_Tp> && !is_void_v<_Tp> && (sizeof(_Tp) > 0), int> = 0>
[[nodiscard]] _CCCL_API constexpr _Tp* midpoint(_Tp* __a, _Tp* __b) noexcept
{
  return __a + ::cuda::std::midpoint(ptrdiff_t(0), __b - __a);
}

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr int __sign(_Tp __val)
{
  return (_Tp(0) < __val) - (__val < _Tp(0));
}

template <typename _Fp>
[[nodiscard]] _CCCL_API constexpr _Fp __fp_abs(_Fp __f)
{
  return __f >= 0 ? __f : -__f;
}

template <class _Fp>
[[nodiscard]] _CCCL_API constexpr enable_if_t<is_floating_point_v<_Fp>, _Fp> midpoint(_Fp __a, _Fp __b) noexcept
{
  constexpr _Fp __lo = numeric_limits<_Fp>::min() * 2;
  constexpr _Fp __hi = numeric_limits<_Fp>::max() / 2;
  return ::cuda::std::__fp_abs(__a) <= __hi && ::cuda::std::__fp_abs(__b) <= __hi
         ? // typical case: overflow is impossible
           (__a + __b) / 2
         : // always correctly rounded
           ::cuda::std::__fp_abs(__a) < __lo ? __a + __b / 2 : // not safe to halve a
             ::cuda::std::__fp_abs(__b) < __lo
             ? __a / 2 + __b
             : // not safe to halve b
             __a / 2 + __b / 2; // otherwise correctly rounded
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___NUMERIC_MIDPOINT_H
