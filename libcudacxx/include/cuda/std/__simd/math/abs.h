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

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/abs.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__internal/features.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__simd/math/common.h>

#if _CCCL_HAS_SIMD_VABSDIFF()
#  include <cuda/__simd/abs_diff.h>
#  include <cuda/std/__cccl/cuda_capabilities.h>
#  include <cuda/std/__simd/reductions.h>

#  include <nv/target>
#endif // _CCCL_HAS_SIMD_VABSDIFF()

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

#if _CCCL_HAS_SIMD_VABSDIFF()

template <typename _Vec>
[[nodiscard]] _CCCL_DEVICE_API inline _Vec __abs_8bit_x4(const _Vec& __x) noexcept
{
  _CCCL_ASSERT(::cuda::std::simd::none_of(__x == numeric_limits<typename _Vec::value_type>::min()),
               "cuda::std::simd::abs precondition: each element must be greater than the minimum "
               "value");
  // abs(x) = abs_diff(x, 0)
  // __vabs4(x) generates several instructions with SM80/CUDA 12.0
  // The packed absolute difference with 0 generates one VABSDIFF4 instruction
  return __simd_abs_diff_impl(
    __x,
    _Vec{},
    ::cuda::simd::__abs_diff_8bit_operation<typename _Vec::value_type>{},
    static_cast<_Vec*>(nullptr)); // ADL
}

#endif // _CCCL_HAS_SIMD_VABSDIFF()

// signed integral
_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(is_integral_v<_Tp> _CCCL_AND is_signed_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto abs(const basic_vec<_Tp, _Abi>& __x) noexcept
{
  using __vec_t = basic_vec<_Tp, _Abi>;
#if _CCCL_HAS_SIMD_VABSDIFF()
  if constexpr (sizeof(_Tp) == 1)
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      if constexpr (__vec_t::size() >= 4)
      {
        NV_IF_TARGET(NV_IS_DEVICE, (return ::cuda::std::simd::__abs_8bit_x4(__x);))
      }
      else
      {
        _CCCL_DISPATCH_NVVM_LATEST_TARGET(return ::cuda::std::simd::__abs_8bit_x4(__x);)
      }
    }
  }
#endif // _CCCL_HAS_SIMD_VABSDIFF()
  return __vec_t{__simd_abs_generator<__vec_t, __vec_t>{__x}};
}

// floating point
_CCCL_TEMPLATE(typename _Vp)
_CCCL_REQUIRES(__is_simd_math_floating_point_v<_Vp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto abs(const _Vp& __x) noexcept
{
  using __result_t = __deduced_vec_t<_Vp>;
  const __result_t __x_vec(__x);
  return __result_t{__simd_abs_generator<__result_t, __result_t>{__x_vec}};
}

// fabs
_CCCL_TEMPLATE(typename _Vp)
_CCCL_REQUIRES(__is_simd_math_floating_point_v<_Vp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto fabs(const _Vp& __x) noexcept
{
  using __result_t = __deduced_vec_t<_Vp>;
  const __result_t __x_vec(__x);
  return __result_t{__simd_abs_generator<__result_t, __result_t>{__x_vec}};
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

//----------------------------------------------------------------------------------------------------------------------

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using simd::abs;
using simd::fabs;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_MATH_ABS_H
