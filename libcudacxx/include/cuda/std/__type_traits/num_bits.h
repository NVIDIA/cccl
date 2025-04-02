//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_NUM_BITS
#define _LIBCUDACXX___TYPE_TRAITS_NUM_BITS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__complex/is_complex.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/has_unique_object_representation.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/climits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <typename _Tp>
struct _PreventInstantiation
{
  static_assert(__always_false_v<_Tp>);
};

template <typename _Tp, typename = remove_cv_t<_Tp>>
inline constexpr int __num_bits_v = _PreventInstantiation<_Tp>{};

template <typename _Tp>
inline constexpr int __num_bits_v<_Tp,
                                  enable_if_t<(has_unique_object_representations_v<_Tp> || is_floating_point_v<_Tp>
                                               || __is_complex_v<_Tp> || is_same_v<_Tp, bool>)
                                                && !__is_sub_byte_floating_point<_Tp>,
                                              remove_cv_t<_Tp>>> = sizeof(_Tp) * CHAR_BIT;

#if _CCCL_HAS_NVFP16()

template <typename _Tp>
inline constexpr int __num_bits_v<_Tp, __half2> = sizeof(__half2) * CHAR_BIT;

#endif // _CCCL_HAS_NVFP16

#if _CCCL_HAS_NVBF16()

template <typename _Tp>
inline constexpr int __num_bits_v<_Tp, __nv_bfloat162> = sizeof(__nv_bfloat162) * CHAR_BIT;

#endif // _CCCL_HAS_NVBF16

#if _CCCL_HAS_NVFP6_E3M2()

template <typename _Tp>
inline constexpr int __num_bits_v<_Tp, __nv_fp6_e3m2> = 6;

#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP6_E2M3()

template <typename _Tp>
inline constexpr int __num_bits_v<_Tp, __nv_fp6_e2m3> = 6;

#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()

template <typename _Tp>
inline constexpr int __num_bits_v<_Tp, __nv_fp4_e2m1> = 4;

#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_FLOAT128()

template <typename _Tp>
inline constexpr int __num_bits_v<_Tp, __float128> = sizeof(__float128) * CHAR_BIT;

#endif // _CCCL_HAS_FLOAT128()

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_NUM_BITS
