//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_MIN_H
#define _CUDA_STD___ALGORITHM_MIN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/comp.h>
#include <cuda/std/__algorithm/comp_ref_type.h>
#include <cuda/std/__algorithm/min_element.h>
#include <cuda/std/__cmath/min_max.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/initializer_list>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Compare>
[[nodiscard]] _CCCL_API constexpr const _Tp& min(const _Tp& __a, const _Tp& __b, _Compare __comp)
{
  return __comp(__b, __a) ? __b : __a;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp>
[[nodiscard]] _CCCL_API constexpr const _Tp& min(const _Tp& __a, const _Tp& __b)
{
  return __b < __a ? __b : __a;
}

template <class _Tp, class _Compare>
[[nodiscard]] _CCCL_API constexpr _Tp min(initializer_list<_Tp> __t, _Compare __comp)
{
  return *::cuda::std::__min_element<__comp_ref_type<_Compare>>(__t.begin(), __t.end(), __comp);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp min(initializer_list<_Tp> __t)
{
  return *::cuda::std::min_element(__t.begin(), __t.end(), __less{});
}

//! @brief Internal version of min that works with values instead of references. Can be used with extended arithmetic
//!        types. Should be preferred as it produces better optimized code with nvcc (see nvbug 5455679).
template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __min(_Tp __a, _Tp __b) noexcept
{
  static_assert(__is_extended_arithmetic_v<_Tp>);
  if constexpr (is_integral_v<_Tp>)
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      if constexpr (sizeof(_Tp) < sizeof(int))
      {
        using _Up = __make_nbit_int_t<32, is_signed_v<_Tp>>;
        return static_cast<_Tp>(::cuda::std::__min(static_cast<_Up>(__a), static_cast<_Up>(__b)));
      }
      else if constexpr (sizeof(_Tp) <= sizeof(long long))
      {
        return ::min(__a, __b);
      }
    }
    return (__b < __a) ? __b : __a;
  }
  else
  {
    return ::cuda::std::fmin(__a, __b);
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_MIN_H
