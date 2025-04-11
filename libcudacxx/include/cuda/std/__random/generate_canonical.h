//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANDOM_GENERATE_CANONICAL_H
#define _LIBCUDACXX___RANDOM_GENERATE_CANONICAL_H

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

_CCCL_PUSH_MACROS

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// generate_canonical
_CCCL_EXEC_CHECK_DISABLE
template <class _RealType, size_t __bits, class _URNG>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI _RealType generate_canonical(_URNG& __g) noexcept
{
  const size_t __dt    = numeric_limits<_RealType>::digits;
  const size_t __b     = __dt < __bits ? __dt : __bits;
  const size_t __log_r = _CUDA_VSTD::__bit_log2<uint64_t>((_URNG::max)() - (_URNG::min)() + uint64_t(1));
  const size_t __k     = __b / __log_r + (__b % __log_r != 0) + (__b == 0);
  const _RealType __rp = static_cast<_RealType>((_URNG::max)() - (_URNG::min)()) + _RealType(1);
  _RealType __base     = __rp;
  _RealType __sp       = __g() - (_URNG::min)();
  for (size_t __i = 1; __i < __k; ++__i, __base *= __rp)
  {
    __sp += (__g() - (_URNG::min)()) * __base;
  }
  return __sp / __base;
}

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_POP_MACROS

#endif // _LIBCUDACXX___RANDOM_GENERATE_CANONICAL_H
