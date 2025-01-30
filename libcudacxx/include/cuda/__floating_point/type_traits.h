//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FLOATING_POINT_TYPE_TRAITS_H
#define _CUDA___FLOATING_POINT_TYPE_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_STD_VER >= 2017

#  include <cuda/__fwd/fp.h>
#  include <cuda/std/__type_traits/remove_cv.h>

#  if _CCCL_HAS_INCLUDE(<stdfloat>)
#    include <stdfloat>
#  endif

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <size_t _NBits>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool __always_false()
{
  return false;
}

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_standard_floating_point_impl_v = false;

template <>
_CCCL_INLINE_VAR constexpr bool __is_standard_floating_point_impl_v<float> = true;

template <>
_CCCL_INLINE_VAR constexpr bool __is_standard_floating_point_impl_v<double> = true;

template <>
_CCCL_INLINE_VAR constexpr bool __is_standard_floating_point_impl_v<long double> = true;

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_standard_floating_point_v =
  __is_standard_floating_point_impl_v<_CUDA_VSTD::remove_cv_t<_Tp>>;

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_std_extended_floating_point_impl_v = false;

#  if __STDCPP_FLOAT16_T__ == 1
template <>
_CCCL_INLINE_VAR constexpr bool __is_std_extended_floating_point_impl_v<::std::float16_t> = true;
#  endif // __STDCPP_FLOAT16_T__ == 1

#  if __STDCPP_BFLOAT16_T__ == 1
template <>
_CCCL_INLINE_VAR constexpr bool __is_std_extended_floating_point_impl_v<::std::bfloat16_t> = true;
#  endif // __STDCPP_BFLOAT16_T__ == 1

#  if __STDCPP_FLOAT32_T__ == 1
template <>
_CCCL_INLINE_VAR constexpr bool __is_std_extended_floating_point_impl_v<::std::float32_t> = true;
#  endif // __STDCPP_FLOAT32_T__ == 1

#  if __STDCPP_FLOAT64_T__ == 1
template <>
_CCCL_INLINE_VAR constexpr bool __is_std_extended_floating_point_impl_v<::std::float64_t> = true;
#  endif // __STDCPP_FLOAT64_T__ == 1

#  if __STDCPP_FLOAT128_T__ == 1
template <>
_CCCL_INLINE_VAR constexpr bool __is_std_extended_floating_point_impl_v<::std::float128_t> = true;
#  endif // __STDCPP_FLOAT128_T__ == 1

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_std_extended_floating_point_v =
  __is_std_extended_floating_point_impl_v<_CUDA_VSTD::remove_cv_t<_Tp>>;

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_cuda_extended_floating_point_impl_v = false;

template <class _Config>
_CCCL_INLINE_VAR constexpr bool __is_cuda_extended_floating_point_impl_v<__fp<_Config>> = true;

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_cuda_extended_floating_point_v =
  __is_cuda_extended_floating_point_impl_v<_CUDA_VSTD::remove_cv_t<_Tp>>;

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __fp_is_floating_point_v =
  __is_standard_floating_point_v<_Tp> || __is_std_extended_floating_point_v<_Tp>
  || __is_cuda_extended_floating_point_v<_Tp>;

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CCCL_STD_VER >= 2017

#endif // _CUDA___FLOATING_POINT_TYPE_TRAITS_H
