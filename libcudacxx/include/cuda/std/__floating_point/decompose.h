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

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__floating_point/cuda_fp_types.h>
#include <cuda/std/__floating_point/format.h>
#include <cuda/std/__floating_point/mask.h>
#include <cuda/std/__floating_point/properties.h>
#include <cuda/std/__floating_point/storage.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

//! @brief Returns the unbiased exponent of a floating point number
//! @param __v The floating point number to extract the exponent from
//! This is effectively the same as @c ilogb but no errors are raised and corner cases handled
template <class _Tp>
[[nodiscard]] _CCCL_API _CCCL_FORCEINLINE constexpr int __fp_get_exp(_Tp __v) noexcept
{
  constexpr auto __fmt = __fp_format_of_v<_Tp>;
  using _Storage       = __fp_storage_t<__fmt>;
  const auto __bits =
    static_cast<_Storage>((::cuda::std::__fp_get_storage(__v) & __fp_exp_mask_v<__fmt>) >> __fp_mant_nbits_v<__fmt>);
  static_assert(__fp_exp_nbits_v<__fmt> < sizeof(int) * CHAR_BIT,
                "__fp_get_exp: floating point type with too many exponent bits");
  return static_cast<int>(__bits) - __fp_exp_bias_v<__fmt>;
}

//! @brief Sets the exponent of a floating point number
//! @param __v The floating point number to set the exponent of
//! @param __exp The new exponent
//! @return A floating point number with the same mantissa and sign as @c __v but the new - biased - exponent @c __exp
template <class _Tp>
[[nodiscard]] _CCCL_API _CCCL_FORCEINLINE constexpr _Tp __fp_set_exp(_Tp __v, int __exp) noexcept
{
  constexpr auto __fmt    = __fp_format_of_v<_Tp>;
  using _Storage          = __fp_storage_t<__fmt>;
  const auto __biased_exp = static_cast<_Storage>(__exp + __fp_exp_bias_v<__fmt>);
  return ::cuda::std::__fp_from_storage<_Tp>(static_cast<_Storage>(
    (::cuda::std::__fp_get_storage(__v) & __fp_inv_exp_mask_v<__fmt>)
    | ((__biased_exp << __fp_mant_nbits_v<__fmt>) &__fp_exp_mask_v<__fmt>) ));
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FLOATING_POINT_DECOMPOSE_H
