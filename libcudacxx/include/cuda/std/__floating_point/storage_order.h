//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FLOATING_POINT_STORAGE_ORDER_H
#define _CUDA_STD___FLOATING_POINT_STORAGE_ORDER_H

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

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// Flipping all bits of negative values and setting the sign bit of positive values maps the sign-magnitude encoding to
// an unsigned order, in which -0 is less than +0. Unsigned formats are already ordered.
// The return type is deduced as the storage type of _Tp, which depends on the format.
template <class _Tp>
[[nodiscard]] _CCCL_API constexpr auto __fp_storage_order_key(_Tp __x) noexcept
{
  using _Storage        = __fp_storage_of_t<_Tp>;
  const _Storage __bits = ::cuda::std::__fp_get_storage(__x);
  if constexpr (__fp_is_signed_v<__fp_format_of_v<_Tp>>)
  {
    constexpr _Storage __sign = __fp_sign_mask_of_v<_Tp>;
    constexpr _Storage __all  = static_cast<_Storage>(__sign | __fp_exp_mant_mask_of_v<_Tp>);
    return static_cast<_Storage>((__bits & __sign) ? (~__bits & __all) : (__bits | __sign));
  }
  else
  {
    return __bits;
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FLOATING_POINT_STORAGE_ORDER_H
