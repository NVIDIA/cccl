//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#if !defined(_CUDA_PTX_BFIND_H)
#  define _CUDA_PTX_BFIND_H

#  include <cuda/std/detail/__config>

#  if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#    pragma GCC system_header
#  elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#    pragma clang system_header
#  elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#    pragma system_header
#  endif // no system header

#  include <cuda/__ptx/ptx_dot_variants.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/cstdint>

#  include <nv/target> // __CUDA_MINIMUM_ARCH__ and friends

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

#  if __cccl_ptx_isa >= 200

template <typename _Tp>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t
bfind(_Tp __x, bfind_shift_amount shift_amt = bfind_shift_amount::disable)
{
  _CUDA_VSTD::uint32_t __ret;
  if constexpr (_CUDA_VSTD::is_same_v<_Tp, _CUDA_VSTD::uint32_t>)
  {
    if (shift_amt == bfind_shift_amount::disable)
    {
      asm("bfind.u32 %0, %1;" : "=r"(__ret) : "r"(__x));
    }
    else
    {
      asm("bfind.shiftamt.u32 %0, %1;" : "=r"(__ret) : "r"(__x));
    }
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Tp, _CUDA_VSTD::uint64_t>)
  {
    if (shift_amt == bfind_shift_amount::disable)
    {
      asm("bfind.u64 %0, %1;" : "=r"(__ret) : "l"(__x));
    }
    else
    {
      asm("bfind.shiftamt.u64 %0, %1;" : "=r"(__ret) : "l"(__x));
    }
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Tp, _CUDA_VSTD::int32_t>)
  {
    if (shift_amt == bfind_shift_amount::disable)
    {
      asm("bfind.s32 %0, %1;" : "=r"(__ret) : "r"(__x));
    }
    else
    {
      asm("bfind.shiftamt.s32 %0, %1;" : "=r"(__ret) : "r"(__x));
    }
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Tp, _CUDA_VSTD::int64_t>)
  {
    if (shift_amt == bfind_shift_amount::disable)
    {
      asm("bfind.s64 %0, %1;" : "=r"(__ret) : "l"(__x));
    }
    else
    {
      asm("bfind.shiftamt.s64 %0, %1;" : "=r"(__ret) : "l"(__x));
    }
  }
  else
  {
    static_assert(sizeof(_Tp) != sizeof(_Tp), "Unsupported type");
  }
  return __ret;
}

#  endif // __cccl_ptx_isa >= 200

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_BFIND_H
