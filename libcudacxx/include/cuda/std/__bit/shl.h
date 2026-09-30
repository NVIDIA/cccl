//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___BIT_SHL_H
#define _CUDA_STD___BIT_SHL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/uabs.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/cstdint>

#if _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()
#  include <cuda/__ptx/instructions/shl.h>
#  include <cuda/__ptx/instructions/shr.h>
#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_TEMPLATE(class _Tp, class _Shift)
_CCCL_REQUIRES(__cccl_is_integer_v<_Tp> _CCCL_AND __cccl_is_integer_v<_Shift>)
[[nodiscard]] _CCCL_API constexpr _Tp shl(const _Tp __v, const _Shift __shift) noexcept
{
  constexpr auto __width = uint32_t{__num_bits_v<_Tp>};
  const auto __ushift    = ::cuda::uabs(__shift);

  if constexpr (is_signed_v<_Shift>)
  {
    if (__shift < 0)
    {
#if !_CCCL_TILE_COMPILATION()
      _CCCL_IF_NOT_CONSTEVAL_DEFAULT
      {
        // On device, shr PTX instruction clamps the shift to width, however only 32-bit shifts are supported.
        NV_IF_TARGET(NV_IS_DEVICE, ({
                       if constexpr (sizeof(_Shift) <= sizeof(uint32_t) && sizeof(_Tp) <= sizeof(int64_t))
                       {
                         using _Up = __make_nbit_int_t<sizeof(_Tp) < sizeof(int64_t) ? 32 : 64, is_signed_v<_Tp>>;
                         return static_cast<_Tp>(::cuda::ptx::shr(_Up{__v}, static_cast<uint32_t>(__ushift)));
                       }
                     }))
      }
#endif // !_CCCL_TILE_COMPILATION()
      return (__ushift < __width) ? (__v >> __ushift) : static_cast<_Tp>(::cuda::std::cmp_less(__v, 0) ? -1 : 0);
    }
  }

#if !_CCCL_TILE_COMPILATION() // error: asm statement is unsupported in tile code
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    // On device, shl PTX instruction clamps the shift to width, however only 32-bit shifts are supported.
    NV_IF_TARGET(NV_IS_DEVICE, ({
                   if constexpr (sizeof(_Shift) <= sizeof(uint32_t) && sizeof(_Tp) <= sizeof(int64_t))
                   {
                     using _Up = __make_nbit_int_t<sizeof(_Tp) < sizeof(int64_t) ? 32 : 64, is_signed_v<_Tp>>;
                     return static_cast<_Tp>(::cuda::ptx::shl(_Up{__v}, static_cast<uint32_t>(__ushift)));
                   }
                 }))
  }
#endif // !_CCCL_TILE_COMPILATION()
  return (__ushift < __width) ? static_cast<_Tp>(::cuda::std::__to_unsigned_like(__v) << __ushift) : _Tp{0};
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BIT_SHL_H
