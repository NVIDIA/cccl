//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FLOATING_POINT_LITERALS_H
#define _CUDA___FLOATING_POINT_LITERALS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_STD_VER >= 2017

#  include <cuda/__floating_point/cast.h>
#  include <cuda/__floating_point/fp.h>

// Silence the warning about the use of long double in device code
_CCCL_NV_DIAG_SUPPRESS(20208)

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

namespace fp_literals
{

#  if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
using __construct_type = long double;
#  else // ^^^ !_LIBCUDACXX_HAS_NO_LONG_DOUBLE ^^^ / vvv _LIBCUDACXX_HAS_NO_LONG_DOUBLE vvv
using __construct_type = double;
#  endif // ^^^ _LIBCUDACXX_HAS_NO_LONG_DOUBLE ^^^

_LIBCUDACXX_HIDE_FROM_ABI constexpr fp4_e2m1 operator""_fp4_e2m1(long double __val) noexcept
{
  return fp4_e2m1{static_cast<__construct_type>(__val)};
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr fp6_e2m3 operator""_fp6_e2m3(long double __val) noexcept
{
  return fp6_e2m3{static_cast<__construct_type>(__val)};
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr fp6_e3m2 operator""_fp6_e3m2(long double __val) noexcept
{
  return fp6_e3m2{static_cast<__construct_type>(__val)};
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr fp16 operator""_fp16(long double __val) noexcept
{
  return fp16{static_cast<__construct_type>(__val)};
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator""_bf16(long double __val) noexcept
{
  return bf16{static_cast<__construct_type>(__val)};
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator""_fp32(long double __val) noexcept
{
  return fp32{static_cast<__construct_type>(__val)};
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator""_fp64(long double __val) noexcept
{
  return fp64{static_cast<__construct_type>(__val)};
}
// #  if !defined(_LIBCUDACXX_HAS_NO_INT128)
// _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator""_fp128(long double __val) noexcept
// {
//   return fp128{static_cast<__construct_type>(__val)};
// }
// #  endif // !_LIBCUDACXX_HAS_NO_INT128

_LIBCUDACXX_HIDE_FROM_ABI constexpr fp8_ue4m3 operator""_fp8_ue4m3(long double __val) noexcept
{
  return fp8_ue4m3{static_cast<__construct_type>(__val)};
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr fp8_ue8m0 operator""_fp8_ue8m0(long double __val) noexcept
{
  return fp8_ue8m0{static_cast<__construct_type>(__val)};
}

} // namespace fp_literals

_LIBCUDACXX_END_NAMESPACE_CUDA

_CCCL_NV_DIAG_DEFAULT(20208)

#endif // _CCCL_STD_VER >= 2017

#endif // _CUDA___FLOATING_POINT_LITERALS_H
