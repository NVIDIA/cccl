//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BIT_BITMASK_H
#define _CUDA___BIT_BITMASK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__ptx/instructions/bmsk.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp bitmask(int __start, int __width) noexcept
{
  static_assert(_CUDA_VSTD::__cccl_is_unsigned_integer_v<_Tp>, "bitmask() requires unsigned integer types");
  constexpr auto __digits = _CUDA_VSTD::numeric_limits<_Tp>::digits;
  _CCCL_ASSERT(__width > 0 && __width <= __digits, "width out of range");
  _CCCL_ASSERT(__start >= 0 && __start < __digits, "start position out of range");
  _CCCL_ASSERT(__start + __width <= __digits, "start position + width out of range");
  if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_PROVIDES_SM_70, (return _CUDA_VPTX::bmsk_clamp(__start, __width);))
    }
  }
  if (__width == __digits)
  {
    return ~_Tp{0};
  }
  return (((_Tp{1} << __width) - 1) << __start);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___BIT_BITMASK_H
