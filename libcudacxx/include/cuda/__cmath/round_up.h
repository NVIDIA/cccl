//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_ROUND_UP_H
#define _CUDA___CMATH_ROUND_UP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_STD_VER >= 2017

#  include <cuda/__cmath/ceil_div.h>
#  include <cuda/std/__type_traits/enable_if.h>
#  include <cuda/std/__type_traits/is_enum.h>
#  include <cuda/std/__type_traits/is_integral.h>
#  include <cuda/std/__type_traits/is_signed.h>
#  include <cuda/std/__type_traits/make_unsigned.h>
#  include <cuda/std/__type_traits/underlying_type.h>
#  include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief Round the number \p __a to the next multiple of \p __b
//! @param __a The input number
//! @param __b The multiplicand
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Tp) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_integral, _Up))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(_Tp{} / _Up{})
round_up(const _Tp __a, const _Up __b) noexcept
{
  _CCCL_ASSERT(__b > _Up{0}, "cuda::round_up: 'b' must be positive");
  if constexpr (_CUDA_VSTD::is_signed_v<_Tp>)
  {
    _CCCL_ASSERT(__a >= _Tp{0}, "cuda::round_up: 'a' must be non negative");
  }
  using _Common  = decltype(_Tp{} / _Up{});
  using _UCommon = _CUDA_VSTD::make_unsigned_t<_Common>;
  auto __c1      = static_cast<_Common>(::cuda::ceil_div(__a, __b));
  _CCCL_ASSERT(__c1 <= _CUDA_VSTD::numeric_limits<_Common>::max() / static_cast<_Common>(__b),
               "cuda::round_up: result overflow");
  return static_cast<_Common>(static_cast<_UCommon>(__c1) * static_cast<_UCommon>(__b));
}

//! @brief Round the number \p __a to the next multiple of \p __b
//! @param __a The input number
//! @param __b The multiplicand
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_enum, _Tp) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_integral, _Up))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(_Tp{} / _CUDA_VSTD::underlying_type_t<_Up>{})
round_up(const _Tp __a, const _Up __b) noexcept
{
  return ::cuda::round_up(__a, static_cast<_CUDA_VSTD::underlying_type_t<_Up>>(__b));
}

//! @brief Round the number \p __a to the next multiple of \p __b
//! @param __a The input number
//! @param __b The multiplicand
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Tp) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_enum, _Up))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(_Tp{} / _CUDA_VSTD::underlying_type_t<_Up>{})
round_up(const _Tp __a, const _Up __b) noexcept
{
  return ::cuda::round_up(static_cast<_CUDA_VSTD::underlying_type_t<_Tp>>(__a), __b);
}

//! @brief Round the number \p __a to the next multiple of \p __b
//! @param __a The input number
//! @param __b The multiplicand
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_enum, _Tp) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_enum, _Up))
_CCCL_NODISCARD
_LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(_CUDA_VSTD::underlying_type_t<_Tp>{} / _CUDA_VSTD::underlying_type_t<_Up>{})
round_up(const _Tp __a, const _Up __b) noexcept
{
  return ::cuda::round_up(static_cast<_CUDA_VSTD::underlying_type_t<_Tp>>(__a),
                          static_cast<_CUDA_VSTD::underlying_type_t<_Up>>(__b));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CCCL_STD_VER >= 2017
#endif // _CUDA___CMATH_ROUND_UP_H
