//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANDOM_GENERATE_CANONICAL_H
#define _CUDA_STD___RANDOM_GENERATE_CANONICAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/integral.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// generate_canonical
_CCCL_EXEC_CHECK_DISABLE
template <class _RealType, size_t __bits, class _URng>
[[nodiscard]] _CCCL_API _RealType generate_canonical(_URng& __g) noexcept
{
  constexpr size_t __dt = numeric_limits<_RealType>::digits;
  const size_t __b      = __dt < __bits ? __dt : __bits;
  const size_t __log_r  = ::cuda::std::__bit_log2<uint64_t>((_URng::max) () - (_URng::min) () + uint64_t(1));
  const size_t __k      = __b / __log_r + (__b % __log_r != 0) + (__b == 0);
  const _RealType __rp  = static_cast<_RealType>((_URng::max) () - (_URng::min) ()) + _RealType(1);
  _RealType __base      = __rp;
  _RealType __sp        = __g() - (_URng::min) ();

  _CCCL_PRAGMA_UNROLL_FULL()
  for (size_t __i = 1; __i < __k; ++__i, __base *= __rp)
  {
    __sp += (__g() - (_URng::min) ()) * __base;
  }
  return __sp / __base;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANDOM_GENERATE_CANONICAL_H
