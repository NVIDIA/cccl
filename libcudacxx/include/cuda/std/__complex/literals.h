//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___COMPLEX_LITERALS_H
#define _CUDA_STD___COMPLEX_LITERALS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// gcc < 8 warns about it's extended literals being shadowed by the implementation, so let's just disable the complex
// literals
#if !_CCCL_COMPILER(GCC, <, 8)

#  include <cuda/std/__complex/complex.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wliteral-suffix")
_CCCL_DIAG_SUPPRESS_CLANG("-Wuser-defined-literals")
_CCCL_DIAG_SUPPRESS_NVHPC(lit_suffix_no_underscore)
_CCCL_DIAG_SUPPRESS_MSVC(4455) // literal suffix identifiers that do not start with an underscore are reserved
_CCCL_BEGIN_NV_DIAG_SUPPRESS(2506, 20208) // a user-provided literal suffix must begin with "_",
                                          // long double treated as double

inline namespace literals
{
inline namespace complex_literals
{
_CCCL_API constexpr complex<long double> operator""il(long double __im)
{
  return {0.0l, __im};
}
_CCCL_API constexpr complex<long double> operator""il(unsigned long long __im)
{
  return {0.0l, static_cast<long double>(__im)};
}

_CCCL_API constexpr complex<double> operator""i(long double __im)
{
  return {0.0, static_cast<double>(__im)};
}

_CCCL_API constexpr complex<double> operator""i(unsigned long long __im)
{
  return {0.0, static_cast<double>(__im)};
}

_CCCL_API constexpr complex<float> operator""if(long double __im)
{
  return {0.0f, static_cast<float>(__im)};
}

_CCCL_API constexpr complex<float> operator""if(unsigned long long __im)
{
  return {0.0f, static_cast<float>(__im)};
}
} // namespace complex_literals
} // namespace literals

_CCCL_END_NV_DIAG_SUPPRESS()
_CCCL_DIAG_POP

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(GCC, <, 8)

#endif // _CUDA_STD___COMPLEX_LITERALS_H
