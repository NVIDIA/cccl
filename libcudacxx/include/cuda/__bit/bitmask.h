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

#include <cuda/std/__bit/shl.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/limits>

#if _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()
#  include <cuda/__ptx/instructions/bmsk.h>
#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _Tp = uint32_t>
[[nodiscard]] _CCCL_API constexpr _Tp bitmask(int __start, int __width) noexcept
{
  static_assert(::cuda::std::__cccl_is_unsigned_integer_v<_Tp>, "bitmask() requires unsigned integer types");
  [[maybe_unused]] constexpr auto __digits = ::cuda::std::numeric_limits<_Tp>::digits;
  _CCCL_ASSERT(__width >= 0 && __width <= __digits, "width out of range");
  _CCCL_ASSERT(__start >= 0 && __start <= __digits, "start position out of range");
  _CCCL_ASSERT(__start + __width <= __digits, "start position + width out of range");
#if !_CCCL_TILE_COMPILATION() // error: asm statement is unsupported in tile code
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
    {
      NV_IF_TARGET(NV_PROVIDES_SM_70, (return ::cuda::ptx::bmsk_clamp(__start, __width);))
    }
  }
#endif // !_CCCL_TILE_COMPILATION()
  return ::cuda::std::shl(static_cast<_Tp>(::cuda::std::shl(_Tp{1}, static_cast<unsigned>(__width)) - 1),
                          static_cast<unsigned>(__start));
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___BIT_BITMASK_H
