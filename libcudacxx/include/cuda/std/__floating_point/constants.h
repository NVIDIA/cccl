//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FLOATING_POINT_CONSTANTS_H
#define _CUDA_STD___FLOATING_POINT_CONSTANTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/arithmetic.h>
#include <cuda/std/__floating_point/format.h>
#include <cuda/std/__floating_point/mask.h>
#include <cuda/std/__floating_point/native_type.h>
#include <cuda/std/__floating_point/properties.h>
#include <cuda/std/__floating_point/storage.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// __fp_inf

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API constexpr __fp_storage_t<_Fmt> __fp_inf() noexcept
{
  static_assert(__fp_has_inf_v<_Fmt>, "The format does not support infinity");

  return static_cast<__fp_storage_t<_Fmt>>(__fp_exp_mask_v<_Fmt> | __fp_explicit_bit_mask_v<_Fmt>);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __fp_inf() noexcept
{
  static_assert(__fp_has_inf_v<__fp_format_of_v<_Tp>>, "The format does not support infinity");

#if defined(_CCCL_BUILTIN_HUGE_VALF)
  if constexpr (__fp_is_native_type_v<_Tp>)
  {
    return static_cast<_Tp>(_CCCL_BUILTIN_HUGE_VALF());
  }
  else
#endif // _CCCL_BUILTIN_HUGE_VALF
  {
    return ::cuda::std::__fp_from_storage<_Tp>(::cuda::std::__fp_inf<__fp_format_of_v<_Tp>>());
  }
}

// __fp_nan

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API constexpr __fp_storage_t<_Fmt> __fp_nan() noexcept
{
  static_assert(__fp_has_nan_v<_Fmt>, "The format does not support nan");

  using _Storage = __fp_storage_t<_Fmt>;

  if constexpr (_Fmt == __fp_format::__fp8_nv_e4m3)
  {
    return _Storage(0x7fu);
  }
  else if constexpr (_Fmt == __fp_format::__fp8_nv_e8m0)
  {
    return _Storage(0xffu);
  }
  else
  {
    return static_cast<_Storage>(__fp_exp_mask_v<_Fmt> | __fp_explicit_bit_mask_v<_Fmt>
                                 | (_Storage(1) << (__fp_mant_nbits_v<_Fmt> - 1 - !__fp_has_implicit_bit_v<_Fmt>) ));
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __fp_nan() noexcept
{
  static_assert(__fp_has_nan_v<__fp_format_of_v<_Tp>>, "The format does not support nan");

#if defined(_CCCL_BUILTIN_NANF)
  if constexpr (__fp_is_native_type_v<_Tp>)
  {
    return static_cast<_Tp>(_CCCL_BUILTIN_NANF(""));
  }
  else
#endif // _CCCL_BUILTIN_NANF
  {
    return ::cuda::std::__fp_from_storage<_Tp>(::cuda::std::__fp_nan<__fp_format_of_v<_Tp>>());
  }
}

// __fp_nans

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API constexpr __fp_storage_t<_Fmt> __fp_nans() noexcept
{
  static_assert(__fp_has_nans_v<_Fmt>, "The format does not support nans");

  using _Storage = __fp_storage_t<_Fmt>;

  return static_cast<_Storage>(__fp_exp_mask_v<_Fmt> | __fp_explicit_bit_mask_v<_Fmt>
                               | (_Storage(1) << (__fp_mant_nbits_v<_Fmt> - 2 - !__fp_has_implicit_bit_v<_Fmt>) ));
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __fp_nans() noexcept
{
  constexpr auto __fmt = __fp_format_of_v<_Tp>;

  static_assert(__fp_has_nans_v<__fmt>, "The format does not support nans");

#if defined(_CCCL_BUILTIN_NANS)
  if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format::__binary32)
  {
    return static_cast<_Tp>(_CCCL_BUILTIN_NANSF(""));
  }
  else if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format::__binary64)
  {
    return static_cast<_Tp>(_CCCL_BUILTIN_NANS(""));
  }
#  if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format_of_v<long double>)
  {
    return static_cast<_Tp>(_CCCL_BUILTIN_NANSL(""));
  }
#  endif // _CCCL_HAS_LONG_DOUBLE
#  if defined(_CCCL_BUILTIN_NANFS128)
  else if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format::__binary128)
  {
    return static_cast<_Tp>(_CCCL_BUILTIN_NANFS128(""));
  }
#  endif // _CCCL_BUILTIN_NANFS128
  else
#endif // _CCCL_BUILTIN_NANS
  {
    return ::cuda::std::__fp_from_storage<_Tp>(::cuda::std::__fp_nans<__fmt>());
  }
}

// __fp_max

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API constexpr __fp_storage_t<_Fmt> __fp_max() noexcept
{
  using _Storage = __fp_storage_t<_Fmt>;

  if constexpr (_Fmt == __fp_format::__fp8_nv_e4m3)
  {
    return _Storage(0x7eu);
  }
  else
  {
    return static_cast<_Storage>(
      (_Storage(__fp_exp_max_v<_Fmt> + __fp_exp_bias_v<_Fmt>) << __fp_mant_nbits_v<_Fmt>) | __fp_mant_mask_v<_Fmt>);
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __fp_max() noexcept
{
  constexpr auto __fmt = __fp_format_of_v<_Tp>;

  if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format::__binary16)
  {
    return static_cast<_Tp>(0x1.ffcp+15f);
  }
  else if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format::__binary32)
  {
    return static_cast<_Tp>(0x1.fffffep+127f);
  }
  else if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format::__binary64)
  {
    return static_cast<_Tp>(0x1.fffffffffffffp+1023);
  }
#if _CCCL_HAS_FLOAT128()
  else if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format::__binary128)
  {
    return static_cast<_Tp>(0x1.ffffffffffffffffffffffffffffp+16383q);
  }
#endif // _CCCL_HAS_FLOAT128()
  else if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format::__bfloat16)
  {
    return static_cast<_Tp>(0x1.fep+127);
  }
  else
  {
    return ::cuda::std::__fp_from_storage<_Tp>(::cuda::std::__fp_max<__fp_format_of_v<_Tp>>());
  }
}

// __fp_min

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API constexpr __fp_storage_t<_Fmt> __fp_min() noexcept
{
  using _Storage = __fp_storage_t<_Fmt>;

  if constexpr (_Fmt == __fp_format::__fp8_nv_e8m0)
  {
    return _Storage{0};
  }
  else
  {
    return static_cast<_Storage>((_Storage{1} << __fp_mant_nbits_v<_Fmt>) | __fp_explicit_bit_mask_v<_Fmt>);
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __fp_min() noexcept
{
  constexpr auto __fmt = __fp_format_of_v<_Tp>;

  if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format::__binary16)
  {
    return static_cast<_Tp>(0x1p-14f);
  }
  else if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format::__binary32)
  {
    return static_cast<_Tp>(0x1p-126f);
  }
  else if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format::__binary64)
  {
    return static_cast<_Tp>(0x1p-1022);
  }
#if _CCCL_HAS_FLOAT128()
  else if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format::__binary128)
  {
    return static_cast<_Tp>(0x1p-16382q);
  }
#endif // _CCCL_HAS_FLOAT128()
  else if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format::__bfloat16)
  {
    return static_cast<_Tp>(0x1p-126f);
  }
  else
  {
    return ::cuda::std::__fp_from_storage<_Tp>(::cuda::std::__fp_min<__fp_format_of_v<_Tp>>());
  }
}

// __fp_denorm_min

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API constexpr __fp_storage_t<_Fmt> __fp_denorm_min() noexcept
{
  static_assert(__fp_has_denorm_v<_Fmt>, "The format does not support denormals");
  return __fp_storage_t<_Fmt>(1);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __fp_denorm_min() noexcept
{
  constexpr auto __fmt = __fp_format_of_v<_Tp>;

  static_assert(__fp_has_denorm_v<__fmt>, "The format does not support denormals");

  if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format::__binary16)
  {
    return static_cast<_Tp>(0x1p-24f);
  }
  else if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format::__binary32)
  {
    return static_cast<_Tp>(0x1p-149f);
  }
  else if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format::__binary64)
  {
    return static_cast<_Tp>(0x1p-1074);
  }
#if _CCCL_HAS_FLOAT128()
  else if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format::__binary128)
  {
    return static_cast<_Tp>(0x1p-16494q);
  }
#endif // _CCCL_HAS_FLOAT128()
  else if constexpr (__fp_is_native_type_v<_Tp> && __fmt == __fp_format::__bfloat16)
  {
    return static_cast<_Tp>(0x1p-133f);
  }
  else
  {
    return ::cuda::std::__fp_from_storage<_Tp>(::cuda::std::__fp_denorm_min<__fmt>());
  }
}

// __fp_lowest

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API constexpr __fp_storage_t<_Fmt> __fp_lowest() noexcept
{
  if constexpr (_Fmt == __fp_format::__fp8_nv_e8m0)
  {
    return ::cuda::std::__fp_min<_Fmt>();
  }
  else
  {
    return ::cuda::std::__fp_neg<_Fmt>(::cuda::std::__fp_max<_Fmt>());
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __fp_lowest() noexcept
{
  if constexpr (__fp_is_native_type_v<_Tp>)
  {
    return ::cuda::std::__fp_neg(::cuda::std::__fp_max<_Tp>());
  }
  else
  {
    return ::cuda::std::__fp_from_storage<_Tp>(::cuda::std::__fp_lowest<__fp_format_of_v<_Tp>>());
  }
}

// __fp_zero

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API constexpr __fp_storage_t<_Fmt> __fp_zero() noexcept
{
  static_assert(_Fmt != __fp_format::__fp8_nv_e8m0, "__fp_zero: __nv_fp8_e8m0 cannot represent zero");
  return __fp_storage_t<_Fmt>(0);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __fp_zero() noexcept
{
  static_assert(__fp_format_of_v<_Tp> != __fp_format::__fp8_nv_e8m0, "__fp_zero: __nv_fp8_e8m0 cannot represent zero");

  if constexpr (__is_std_fp_v<_Tp> || __is_ext_compiler_fp_v<_Tp>)
  {
    return _Tp{};
  }
  else
  {
    return ::cuda::std::__fp_from_storage<_Tp>(__fp_storage_of_t<_Tp>{0});
  }
}

// __fp_one

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API constexpr __fp_storage_t<_Fmt> __fp_one() noexcept
{
  return static_cast<__fp_storage_t<_Fmt>>(__fp_exp_bias_v<_Fmt> << __fp_mant_nbits_v<_Fmt>);
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __fp_one() noexcept
{
  if constexpr (__is_std_fp_v<_Tp> || __is_ext_compiler_fp_v<_Tp>)
  {
    return _Tp{1};
  }
  else
  {
    return ::cuda::std::__fp_from_storage<_Tp>(::cuda::std::__fp_one<__fp_format_of_v<_Tp>>());
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FLOATING_POINT_CONSTANTS_H
