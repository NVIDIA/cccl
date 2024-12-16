// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CSTDLIB_ABS_H
#define _LIBCUDACXX___CSTDLIB_ABS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int abs(int __val) noexcept
{
  return (__val < 0) ? -__val : __val;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr long labs(long __val) noexcept
{
  return (__val < 0l) ? -__val : __val;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr long abs(long __val) noexcept
{
  return _CUDA_VSTD::labs(__val);
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr long long llabs(long long __val) noexcept
{
  return (__val < 0ll) ? -__val : __val;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr long long abs(long long __val) noexcept
{
  return _CUDA_VSTD::llabs(__val);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CSTDLIB_ABS_H
