//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FLOATING_POINT_ARITHMETIC_H
#define _LIBCUDACXX___FLOATING_POINT_ARITHMETIC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/mask.h>
#include <cuda/std/__floating_point/native_type.h>
#include <cuda/std/__floating_point/storage.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __fp_neg(const _Tp& __v) noexcept
{
  if constexpr (__fp_is_native_type_v<_Tp>)
  {
    return -__v;
  }
  else
  {
    const auto __res_storage = _CUDA_VSTD::__fp_get_storage(__v) ^ __fp_sign_mask_of_v<_Tp>;
    return _CUDA_VSTD::__fp_from_storage<_Tp>(static_cast<__fp_storage_of_t<_Tp>>(__res_storage));
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FLOATING_POINT_ARITHMETIC_H
