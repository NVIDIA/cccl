//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___UTILITY_CONVERT_TO_INTEGRAL_H
#define _CUDA_STD___UTILITY_CONVERT_TO_INTEGRAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_scoped_enum.h>
#include <cuda/std/__type_traits/underlying_type.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_API constexpr void __convert_to_integral(...) noexcept;

[[nodiscard]] _CCCL_API constexpr int __convert_to_integral(int __val) noexcept
{
  return __val;
}

[[nodiscard]] _CCCL_API constexpr unsigned __convert_to_integral(unsigned __val) noexcept
{
  return __val;
}

[[nodiscard]] _CCCL_API constexpr long __convert_to_integral(long __val) noexcept
{
  return __val;
}

[[nodiscard]] _CCCL_API constexpr unsigned long __convert_to_integral(unsigned long __val) noexcept
{
  return __val;
}

[[nodiscard]] _CCCL_API constexpr long long __convert_to_integral(long long __val) noexcept
{
  return __val;
}

[[nodiscard]] _CCCL_API constexpr unsigned long long __convert_to_integral(unsigned long long __val) noexcept
{
  return __val;
}

#if _CCCL_HAS_INT128()
[[nodiscard]] _CCCL_API constexpr __int128_t __convert_to_integral(__int128_t __val) noexcept
{
  return __val;
}

[[nodiscard]] _CCCL_API constexpr __uint128_t __convert_to_integral(__uint128_t __val) noexcept
{
  return __val;
}
#endif // _CCCL_HAS_INT128()

_CCCL_TEMPLATE(class _Fp)
_CCCL_REQUIRES(is_floating_point_v<_Fp>)
[[nodiscard]] _CCCL_API constexpr long long __convert_to_integral(_Fp __val) noexcept
{
  return static_cast<long long>(__val);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES((!is_scoped_enum_v<_Tp>) )
[[nodiscard]] _CCCL_API constexpr decltype(((underlying_type_t<_Tp>) 1) + 0) __convert_to_integral(_Tp __val) noexcept
{
  return __val;
}

template <class _Int>
struct __avoid_promotions;

template <class _Int>
[[nodiscard]] _CCCL_API constexpr _Int __convert_to_integral(__avoid_promotions<_Int> __self) noexcept
{
  return __self.__data;
}

template <class _Tp, class _Integral = decltype(::cuda::std::__convert_to_integral(::cuda::std::declval<_Tp>()))>
inline constexpr bool __is_convertible_to_integral_v = true;
template <class _Tp>
inline constexpr bool __is_convertible_to_integral_v<_Tp, void> = false;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___UTILITY_CONVERT_TO_INTEGRAL_H
