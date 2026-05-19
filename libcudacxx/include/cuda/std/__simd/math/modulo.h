//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_MATH_MODULO_H
#define _CUDA_STD___SIMD_MATH_MODULO_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/modulo.h>
#include <cuda/std/__cmath/remainder.h>
#include <cuda/std/__simd/math/common.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/array>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

_CCCL_SIMD_MATH_BINARY_GENERATOR(fmod, fmod);
_CCCL_SIMD_MATH_BINARY_GENERATOR(remainder, remainder);

_CCCL_SIMD_MATH_BINARY_FUNCTION(fmod, fmod, )
_CCCL_SIMD_MATH_BINARY_FUNCTION(remainder, remainder, )

//----------------------------------------------------------------------------------------------------------------------
// remquo

template <typename _Result, typename _Vp0, typename _Vp1>
struct __simd_remquo_generator
{
  using __result_t = typename _Result::value_type;

  const _Vp0& __x_;
  const _Vp1& __y_;
  array<int, _Result::__usize>& __quotients_;

  template <typename _Ip>
  [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Ip) const noexcept
  {
    int __quotient           = 0;
    const auto __remquo      = ::cuda::std::remquo(__x_[_Ip::value], __y_[_Ip::value], &__quotient);
    const auto __result      = static_cast<__result_t>(__remquo);
    __quotients_[_Ip::value] = __quotient;
    return __result;
  }
};

template <typename _Result, typename _Vp0, typename _Vp1>
[[nodiscard]] _CCCL_API _Result
__simd_remquo_impl(const _Vp0& __x, const _Vp1& __y, rebind_t<int, _Result>* __quo) noexcept
{
  array<int, _Result::__usize> __quotients{};
  const _Result __values{__simd_remquo_generator<_Result, _Vp0, _Vp1>{__x, __y, __quotients}};
  *__quo = rebind_t<int, _Result>{__quotients};
  return __values;
}

_CCCL_TEMPLATE(typename _Vp0, typename _Vp1, typename _Result = __simd_math_result_t<_Vp0, _Vp1>)
_CCCL_REQUIRES(__is_simd_math_v<_Result, _Vp0, _Vp1>)
[[nodiscard]] _CCCL_API _Result remquo(const _Vp0& __x, const _Vp1& __y, rebind_t<int, _Result>* __quo) noexcept
{
  const _Result __x_vec = __x;
  const _Result __y_vec = __y;
  return ::cuda::std::simd::__simd_remquo_impl<_Result, _Result, _Result>(__x_vec, __y_vec, __quo);
}

//----------------------------------------------------------------------------------------------------------------------
// modf

template <typename _Vp>
struct __simd_modf_generator
{
  using __result_t = typename _Vp::value_type;

  const _Vp& __x_;
  array<__result_t, _Vp::__usize>& __integrals_;

  template <typename _Ip>
  [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Ip) const noexcept
  {
    __result_t __integral    = 0;
    auto __modf              = ::cuda::std::modf(__x_[_Ip::value], &__integral);
    const auto __result      = static_cast<__result_t>(__modf);
    __integrals_[_Ip::value] = __integral;
    return __result;
  }
};

// modf is the only function that doesn't have constraints on the type, even if modf is only defined for floating point
// types.

template <typename _Tp, typename _Abi>
[[nodiscard]] _CCCL_API basic_vec<_Tp, _Abi>
modf(const type_identity_t<basic_vec<_Tp, _Abi>>& __x, basic_vec<_Tp, _Abi>* __iptr) noexcept
{
  using _Vp = basic_vec<_Tp, _Abi>;
  array<_Tp, _Vp::__usize> __integrals{};
  const _Vp __values{__simd_modf_generator<_Vp>{__x, __integrals}};
  *__iptr = _Vp{__integrals};
  return __values;
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

//----------------------------------------------------------------------------------------------------------------------

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using simd::fmod;
using simd::modf;
using simd::remainder;
using simd::remquo;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_MATH_MODULO_H
