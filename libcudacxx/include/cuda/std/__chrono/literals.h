// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHRONO_LITERALS_H
#define _CUDA_STD___CHRONO_LITERALS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__chrono/day.h>
#include <cuda/std/__chrono/year.h>

#include <cuda/std/__cccl/prologue.h>

// Silence NVCC warnings `long double` arising from chrono floating pointer
// user-defined literals which are defined in terms of `long double`.

// FIXME: There is currently no way to disable this diagnostic in a fine-grained
// fashion; if you include this header, the diagnostic will be suppressed
// throughout the translation unit. The alternative is loosing (conforming)
// chrono user-defined literals; this seems like the lesser of two evils, so...
_CCCL_BEGIN_NV_DIAG_SUPPRESS(cuda_demote_unsupported_floating_point)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wliteral-suffix")
_CCCL_DIAG_SUPPRESS_CLANG("-Wuser-defined-literals")
_CCCL_DIAG_SUPPRESS_NVHPC(lit_suffix_no_underscore)
_CCCL_DIAG_SUPPRESS_MSVC(4455) // literal suffix identifiers that do not start with an underscore are reserved
_CCCL_BEGIN_NV_DIAG_SUPPRESS(2506) // a user-provided literal suffix must begin with "_"

// Suffixes for duration literals [time.duration.literals]
inline namespace literals
{
inline namespace chrono_literals
{
_CCCL_API constexpr chrono::hours operator""h(unsigned long long __h) noexcept
{
  return chrono::hours(static_cast<chrono::hours::rep>(__h));
}

_CCCL_API constexpr chrono::duration<double, ratio<3600, 1>> operator""h(long double __h) noexcept
{
  return chrono::duration<double, ratio<3600, 1>>(__h);
}

_CCCL_API constexpr chrono::minutes operator""min(unsigned long long __m) noexcept
{
  return chrono::minutes(static_cast<chrono::minutes::rep>(__m));
}

_CCCL_API constexpr chrono::duration<double, ratio<60, 1>> operator""min(long double __m) noexcept
{
  return chrono::duration<double, ratio<60, 1>>(__m);
}

_CCCL_API constexpr chrono::seconds operator""s(unsigned long long __s) noexcept
{
  return chrono::seconds(static_cast<chrono::seconds::rep>(__s));
}

_CCCL_API constexpr chrono::duration<double> operator""s(long double __s) noexcept
{
  return chrono::duration<double>(__s);
}

_CCCL_API constexpr chrono::milliseconds operator""ms(unsigned long long __ms) noexcept
{
  return chrono::milliseconds(static_cast<chrono::milliseconds::rep>(__ms));
}

_CCCL_API constexpr chrono::duration<double, milli> operator""ms(long double __ms) noexcept
{
  return chrono::duration<double, milli>(__ms);
}

_CCCL_API constexpr chrono::microseconds operator""us(unsigned long long __us) noexcept
{
  return chrono::microseconds(static_cast<chrono::microseconds::rep>(__us));
}

_CCCL_API constexpr chrono::duration<double, micro> operator""us(long double __us) noexcept
{
  return chrono::duration<double, micro>(__us);
}

_CCCL_API constexpr chrono::nanoseconds operator""ns(unsigned long long __ns) noexcept
{
  return chrono::nanoseconds(static_cast<chrono::nanoseconds::rep>(__ns));
}

_CCCL_API constexpr chrono::duration<double, nano> operator""ns(long double __ns) noexcept
{
  return chrono::duration<double, nano>(__ns);
}

#if _LIBCUDACXX_HAS_CXX20_CHRONO_LITERALS()
_CCCL_API constexpr chrono::day operator""d(unsigned long long __d) noexcept
{
  return chrono::day(static_cast<unsigned>(__d));
}

_CCCL_API constexpr chrono::year operator""y(unsigned long long __y) noexcept
{
  return chrono::year(static_cast<int>(__y));
}
#endif // _LIBCUDACXX_HAS_CXX20_CHRONO_LITERALS()
} // namespace chrono_literals
} // namespace literals

namespace chrono
{ // hoist the literals into namespace cuda::std::chrono
using namespace literals::chrono_literals;
} // namespace chrono

_CCCL_END_NV_DIAG_SUPPRESS()
_CCCL_DIAG_POP

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHRONO_LITERALS_H
