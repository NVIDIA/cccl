//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FLOATING_POINT_CONV_RANK_ORDER_H
#define _CUDA___FLOATING_POINT_CONV_RANK_ORDER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_STD_VER >= 2017

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

#  include <cuda/__floating_point/config.h>
#  include <cuda/__floating_point/type_traits.h>
#  include <cuda/std/__type_traits/is_same.h>

enum class __fp_conv_rank_order
{
  __unordered,
  __greater,
  __equal,
  __less,
};

template <class _To, class _From>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __fp_conv_rank_order __fp_make_conv_rank_order()
{
  using _FromConfig = __fp_make_config_from_t<_From>;
  using _ToConfig   = __fp_make_config_from_t<_To>;

  if constexpr (!_CUDA_VSTD::is_same_v<_FromConfig, __fp_invalid_config>
                && !_CUDA_VSTD::is_same_v<_ToConfig, __fp_invalid_config>)
  {
    if constexpr (_ToConfig::__is_signed == _FromConfig::__is_signed)
    {
      if constexpr (_ToConfig::__exp_nbits == _FromConfig::__exp_nbits
                    && _ToConfig::__mant_nbits == _FromConfig::__mant_nbits)
      {
#  if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
        // If the representations are the same, long double always has the higher subrank
        if constexpr (_CUDA_VSTD::is_same_v<_ToConfig, __fp_long_double_config>
                      && !_CUDA_VSTD::is_same_v<_FromConfig, __fp_long_double_config>)
        {
          return __fp_conv_rank_order::__greater;
        }
        else if constexpr (!_CUDA_VSTD::is_same_v<_ToConfig, __fp_long_double_config>
                           && _CUDA_VSTD::is_same_v<_FromConfig, __fp_long_double_config>)
        {
          return __fp_conv_rank_order::__less;
        }
#  endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
        return __fp_conv_rank_order::__equal;
      }
      else if constexpr (_ToConfig::__exp_nbits >= _FromConfig::__exp_nbits
                         && _ToConfig::__mant_nbits >= _FromConfig::__mant_nbits)
      {
        return __fp_conv_rank_order::__greater;
      }
      else if constexpr (_ToConfig::__exp_nbits <= _FromConfig::__exp_nbits
                         && _ToConfig::__mant_nbits <= _FromConfig::__mant_nbits)
      {
        return __fp_conv_rank_order::__less;
      }
    }
  }

  return __fp_conv_rank_order::__unordered;
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CCCL_STD_VER >= 2017

#endif // _CUDA___FLOATING_POINT_CONV_RANK_ORDER_H
