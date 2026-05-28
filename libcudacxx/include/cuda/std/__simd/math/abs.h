//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_MATH_ABS_H
#define _CUDA_STD___SIMD_MATH_ABS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/abs.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__simd/math/common.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

template <typename _Result, typename _Vp>
struct __simd_abs_generator
{
  using __result_t = typename _Result::value_type;

  const _Vp& __x_;

  template <typename _Ip>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __result_t operator()(_Ip) const noexcept
  {
    const auto __x = __x_[_Ip::value];
    if constexpr (is_unsigned_v<__result_t>)
    {
      return __x;
    }
    else if constexpr (is_integral_v<__result_t>)
    {
      _CCCL_ASSERT(__x >= -numeric_limits<__result_t>::max(),
                   "cuda::std::simd::abs precondition: each element must be greater than the minimum value");
      return (__x < __result_t{0}) ? static_cast<__result_t>(-__x) : __x;
    }
    else
    {
      return ::cuda::std::fabs(__x);
    }
  }
};

// signed integral
_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(is_integral_v<_Tp> _CCCL_AND is_signed_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto abs(const basic_vec<_Tp, _Abi>& __x) noexcept
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__simd_abs_generator<__vec_t, __vec_t>{__x}};
}

// floating point
_CCCL_TEMPLATE(typename _Vp)
_CCCL_REQUIRES(__is_math_floating_point_v<_Vp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto abs(const _Vp& __x) noexcept
{
  using __result_t = __deduced_vec_t<_Vp>;
  return __result_t{__simd_abs_generator<__result_t, _Vp>{__x}};
}

// fabs
_CCCL_TEMPLATE(typename _Vp)
_CCCL_REQUIRES(__is_math_floating_point_v<_Vp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto fabs(const _Vp& __x) noexcept
{
  using __result_t = __deduced_vec_t<_Vp>;
  return __result_t{__simd_abs_generator<__result_t, _Vp>{__x}};
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

//----------------------------------------------------------------------------------------------------------------------

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using simd::abs;
using simd::fabs;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_MATH_ABS_H
