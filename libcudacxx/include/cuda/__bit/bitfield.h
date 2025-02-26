//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BIT_BITFILED_INSERT_EXTRACT_H
#define _CUDA___BIT_BITFILED_INSERT_EXTRACT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__ptx/instructions/bmsk.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

#if _CCCL_HAS_CUDA_COMPILER

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI uint32_t __bfi(uint32_t __value, int __start, int __width) noexcept
{
  uint32_t __ret;
  asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(__ret) : "r"(0xFFFFFFFF), "r"(__value), "r"(__start), "r"(__width));
  return __ret;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI uint32_t __bfe(uint32_t __value, int __start, int __width) noexcept
{
  uint32_t __ret;
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(__ret) : "r"(__value), "r"(__start), "r"(__width));
  return __ret;
}

#endif // _CCCL_HAS_CUDA_COMPILER

template <typename _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp
bitfield_insert(const _Tp __value, int __start, int __width = 1) noexcept
{
  static_assert(_CUDA_VSTD::__cccl_is_unsigned_integer_v<_Tp>, "bitfield_insert() requires unsigned integer types");
  constexpr auto __digits = _CUDA_VSTD::numeric_limits<_Tp>::digits;
  _CCCL_ASSERT(__width > 0 && __width <= __digits, "width out of range");
  _CCCL_ASSERT(__start >= 0 && __start < __digits, "start position out of range");
  _CCCL_ASSERT(__start + __width <= __digits, "start position + width out of range");
  if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      // clang-format off
      NV_DISPATCH_TARGET(
        NV_PROVIDES_SM_70, (return __value | _CUDA_VPTX::bmsk_clamp(__start, __width);),
        NV_IS_DEVICE,      (return ::cuda::__bfi(static_cast<uint32_t>(__value), __start, __width);))
      // clang-format on
    }
  }
  if (__width == __digits)
  {
    return ~_Tp{0};
  }
  auto __mask = (_Tp{1} << __width) - 1;
  return __value | (__mask << __start);
}

template <typename _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp
bitfield_extract(const _Tp __value, int __start, int __width = 1) noexcept
{
  static_assert(_CUDA_VSTD::__cccl_is_unsigned_integer_v<_Tp>, "bitfield_extract() requires unsigned integer types");
  constexpr auto __digits = _CUDA_VSTD::numeric_limits<_Tp>::digits;
  _CCCL_ASSERT(__width > 0 && __width <= __digits, "width out of range");
  _CCCL_ASSERT(__start >= 0 && __start < __digits, "start position out of range");
  _CCCL_ASSERT(__start + __width <= __digits, "start position + width out of range");
  if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      // clang-format off
      NV_DISPATCH_TARGET(
        NV_PROVIDES_SM_70, (return __value & _CUDA_VPTX::bmsk_clamp(__start, __width);),
        NV_IS_DEVICE,      (return ::cuda::__bfe(static_cast<uint32_t>(__value), __start, __width);))
      // clang-format on
    }
  }
  if (__width == __digits)
  {
    return __value;
  }
  auto __mask = (_Tp{1} << __width) - 1;
  return __value & (__mask << __start);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___BIT_BITFILED_INSERT_EXTRACT_H
