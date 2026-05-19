//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_MATH_MANIPULATION_H
#define _CUDA_STD___SIMD_MATH_MANIPULATION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/copysign.h>
#include <cuda/std/__cmath/exponential_functions.h>
#include <cuda/std/__cmath/logarithms.h>
#include <cuda/std/__cmath/rounding_functions.h>
#include <cuda/std/__simd/math/common.h>
#include <cuda/std/array>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

_CCCL_SIMD_MATH_UNARY_GENERATOR(ilogb);
_CCCL_SIMD_MATH_UNARY_FUNCTION(logb, constexpr)

_CCCL_SIMD_MATH_BINARY_GENERATOR(ldexp, ldexp);
_CCCL_SIMD_MATH_BINARY_GENERATOR(scalbn, scalbn);
_CCCL_SIMD_MATH_BINARY_GENERATOR(scalbln, scalbln);
_CCCL_SIMD_MATH_BINARY_GENERATOR(nextafter, nextafter);
_CCCL_SIMD_MATH_BINARY_GENERATOR(copysign, copysign);

_CCCL_SIMD_MATH_UNARY_REBIND_FUNCTION(ilogb, int, constexpr)

_CCCL_SIMD_MATH_BINARY_FUNCTION(nextafter, nextafter, )
_CCCL_SIMD_MATH_BINARY_FUNCTION(copysign, copysign, constexpr)

//----------------------------------------------------------------------------------------------------------------------
// ldexp, scalbn, scalbln

#define _CCCL_SIMD_MATH_BINARY_REBIND_FUNCTION(_NAME, _Tp, _CONSTEXPR)                                         \
  _CCCL_TEMPLATE(typename _Vp, typename _Result = __deduced_vec_t<_Vp>)                                        \
  _CCCL_REQUIRES(__is_math_floating_point_v<_Vp>)                                                              \
  [[nodiscard]] _CCCL_API _CONSTEXPR _Result _NAME(const _Vp& __x, const rebind_t<_Tp, _Result>& __y) noexcept \
  {                                                                                                            \
    return _Result{__simd_##_NAME##_generator<_Result, _Vp, rebind_t<_Tp, _Result>>{__x, __y}};                \
  }

_CCCL_SIMD_MATH_BINARY_REBIND_FUNCTION(ldexp, int, )
_CCCL_SIMD_MATH_BINARY_REBIND_FUNCTION(scalbn, int, )
_CCCL_SIMD_MATH_BINARY_REBIND_FUNCTION(scalbln, long, )

#undef _CCCL_SIMD_MATH_BINARY_REBIND_FUNCTION

// frexp

template <typename _Result, typename _Vp>
struct __simd_frexp_generator
{
  using __result_t = typename _Result::value_type;

  const _Vp& __x_;
  array<int, _Result::__usize>& __exponents_;

  template <typename _Ip>
  [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Ip) const noexcept
  {
    int __exponent           = 0;
    const auto __result      = static_cast<__result_t>(::cuda::std::frexp(__x_[_Ip::value], &__exponent));
    __exponents_[_Ip::value] = __exponent;
    return __result;
  }
};

_CCCL_TEMPLATE(typename _Vp, typename _Result = __deduced_vec_t<_Vp>)
_CCCL_REQUIRES(__is_math_floating_point_v<_Vp>)
[[nodiscard]] _CCCL_API _Result frexp(const _Vp& __x, rebind_t<int, _Result>* __exp) noexcept
{
  using __exp_t = rebind_t<int, _Result>;
  array<int, _Result::__usize> __exponents{};

  const _Result __values{__simd_frexp_generator<_Result, _Vp>{__x, __exponents}};
  *__exp = __exp_t{__exponents};
  return __values;
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

//----------------------------------------------------------------------------------------------------------------------

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using simd::copysign;
using simd::frexp;
using simd::ilogb;
using simd::ldexp;
using simd::logb;
using simd::nextafter;
using simd::scalbln;
using simd::scalbn;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_MATH_MANIPULATION_H
