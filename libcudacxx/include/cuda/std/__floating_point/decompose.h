//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FLOATING_POINT_DECOMPOSE_H
#define _CUDA_STD___FLOATING_POINT_DECOMPOSE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/format.h>
#include <cuda/std/__floating_point/mask.h>
#include <cuda/std/__floating_point/properties.h>
#include <cuda/std/__floating_point/storage.h>
#include <cuda/std/climits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API _CCCL_FORCEINLINE constexpr int __fp_get_exp_biased(__fp_storage_t<_Fmt> __v) noexcept
{
  using _Storage    = __fp_storage_t<_Fmt>;
  const auto __bits = static_cast<_Storage>((__v & __fp_exp_mask_v<_Fmt>) >> __fp_mant_nbits_v<_Fmt>);
  static_assert(__fp_exp_nbits_v<_Fmt> < sizeof(int) * CHAR_BIT,
                "__fp_get_exp: floating point type with too many exponent bits");
  return static_cast<int>(__bits);
}

//! @brief Returns the biased exponent of a floating point number
//! @param __v The floating point number to extract the exponent from
//! This is effectively the same as @c ilogb but no errors are raised and corner cases handled
template <class _Tp>
[[nodiscard]] _CCCL_API _CCCL_FORCEINLINE constexpr int __fp_get_exp_biased(_Tp __v) noexcept
{
  return ::cuda::std::__fp_get_exp_biased<__fp_format_of_v<_Tp>>(::cuda::std::__fp_get_storage(__v));
}

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API _CCCL_FORCEINLINE constexpr int __fp_get_exp(__fp_storage_t<_Fmt> __v) noexcept
{
  return ::cuda::std::__fp_get_exp_biased<_Fmt>(__v) - __fp_exp_bias_v<_Fmt>;
}

//! @brief Returns the unbiased exponent of a floating point number
//! @param __v The floating point number to extract the exponent from
//! This is effectively the same as @c ilogb but no errors are raised and corner cases handled
template <class _Tp>
[[nodiscard]] _CCCL_API _CCCL_FORCEINLINE constexpr int __fp_get_exp(_Tp __v) noexcept
{
  return ::cuda::std::__fp_get_exp<__fp_format_of_v<_Tp>>(::cuda::std::__fp_get_storage(__v));
}

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API _CCCL_FORCEINLINE constexpr __fp_storage_t<_Fmt>
__fp_set_exp_biased(__fp_storage_t<_Fmt> __v, int __exp) noexcept
{
  using _Storage = __fp_storage_t<_Fmt>;
  auto __result  = static_cast<_Storage>(
    (__v & __fp_inv_exp_mask_v<_Fmt>)
    | ((static_cast<_Storage>(__exp) << __fp_mant_nbits_v<_Fmt>) &__fp_exp_mask_v<_Fmt>) );

  // if the type has explicit bit, it must be set to 1 if exponent is greater than 1 and to 0 otherwise.
  if constexpr (!__fp_has_implicit_bit_v<_Fmt>)
  {
    if (__exp == 0)
    {
      __result &= ~__fp_explicit_bit_mask_v<_Fmt>;
    }
    else
    {
      __result |= __fp_explicit_bit_mask_v<_Fmt>;
    }
  }
  return __result;
}

//! @brief Sets the already biased exponent of a floating point number
//! @param __v The floating point number to set the exponent of
//! @param __exp The new exponent
//! @note This function also sets the explicit bit if necessary.
//! @return A floating point number with the same mantissa and sign as @c __v but the new - exponent @c __exp
template <class _Tp>
[[nodiscard]] _CCCL_API _CCCL_FORCEINLINE constexpr _Tp __fp_set_exp_biased(_Tp __v, int __exp) noexcept
{
  return ::cuda::std::__fp_from_storage<_Tp>(
    ::cuda::std::__fp_set_exp_biased<__fp_format_of_v<_Tp>>(::cuda::std::__fp_get_storage(__v), __exp));
}

template <__fp_format _Fmt>
[[nodiscard]] _CCCL_API _CCCL_FORCEINLINE constexpr __fp_storage_t<_Fmt>
__fp_set_exp(__fp_storage_t<_Fmt> __v, int __exp) noexcept
{
  return ::cuda::std::__fp_set_exp_biased<_Fmt>(__v, __exp + __fp_exp_bias_v<_Fmt>);
}

//! @brief Sets the exponent of a floating point number
//! @param __v The floating point number to set the exponent of
//! @param __exp The new exponent
//! @return A floating point number with the same mantissa and sign as @c __v but the new - biased - exponent @c __exp
template <class _Tp>
[[nodiscard]] _CCCL_API _CCCL_FORCEINLINE constexpr _Tp __fp_set_exp(_Tp __v, int __exp) noexcept
{
  return ::cuda::std::__fp_from_storage<_Tp>(
    ::cuda::std::__fp_set_exp<__fp_format_of_v<_Tp>>(::cuda::std::__fp_get_storage(__v), __exp));
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FLOATING_POINT_DECOMPOSE_H
