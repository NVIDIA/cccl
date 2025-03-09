//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___INTERNAL_FP_COMMON_TYPE_H
#define _LIBCUDACXX___INTERNAL_FP_COMMON_TYPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__internal/fp/conversion_rank_order.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integral.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Lhs, class _Rhs>
_LIBCUDACXX_HIDE_FROM_ABI constexpr auto __fp_common_type_impl()
{
  constexpr auto __order = __fp_conv_rank_order_v<_Lhs, _Rhs>;

  if constexpr (__order == __fp_conv_rank_order::__greater)
  {
    return _Lhs{};
  }
  else if constexpr (__order != __fp_conv_rank_order::__unordered)
  {
    return _Rhs{};
  }
  else
  {
    static_assert(__always_false_v<_Lhs>, "No common type");
  }
}

template <class _Lhs, class _Rhs>
using __fp_common_type_t = decltype(__fp_common_type_impl<_Lhs, _Rhs>());

template <class _Lhs, class _Rhs>
using __fp_int_ext_common_type_t =
  __fp_common_type_t<conditional_t<_CCCL_TRAIT(is_integral, _Lhs), double, _Lhs>,
                     conditional_t<_CCCL_TRAIT(is_integral, _Rhs), double, _Rhs>>;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___INTERNAL_FP_COMMON_TYPE_H
