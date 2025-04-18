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

#include <cuda/std/__floating_point/nvfp_types.h>
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
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr int __num_bits_impl() noexcept
{
  if constexpr (false)
  {
  }
#if _CCCL_HAS_NVFP16()
  else if constexpr (is_same_v<_Tp, __half2>)
  {
    return sizeof(_Tp) * CHAR_BIT;
  }
#endif // _CCCL_HAS_NVFP16
#if _CCCL_HAS_NVBF16()
  else if constexpr (is_same_v<_Tp, __nv_bfloat162>)
  {
    return sizeof(_Tp) * CHAR_BIT;
  }
#endif // _CCCL_HAS_NVBF16
#if _CCCL_HAS_NVFP8_E8M0()
  else if constexpr (is_same_v<_Tp, __nv_fp8_e8m0>)
  {
    return 8;
  }
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E3M2()
  else if constexpr (is_same_v<_Tp, __nv_fp6_e3m2>)
  {
    return 6;
  }
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP6_E2M3()
  else if constexpr (is_same_v<_Tp, __nv_fp6_e2m3>)
  {
    return 6;
  }
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP4_E2M1()
  else if constexpr (is_same_v<_Tp, __nv_fp4_e2m1>)
  {
    return 4;
  }
#endif // _CCCL_HAS_NVFP4_E2M1()
#if _CCCL_HAS_FLOAT128()
  else if constexpr (is_same_v<_Tp, __float128>)
  {
    return sizeof(_Tp) * CHAR_BIT;
  }
#endif // _CCCL_HAS_FLOAT128()
  else if constexpr (has_unique_object_representations_v<_Tp> || is_floating_point_v<_Tp> || is_same_v<_Tp, bool>)
  {
    return sizeof(_Tp) * CHAR_BIT;
  }
  else
  {
    static_assert(__always_false_v<_Tp>, "unsupported type");
    //_CCCL_UNREACHABLE();
    return 0;
  }
}

template <typename _Tp>
inline constexpr int __num_bits_v = __num_bits_impl<remove_cv_t<_Tp>>();

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_NUM_BITS
