//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_ENDIAN_H
#define _LIBCUDACXX___BIT_ENDIAN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

enum class endian
{
  little = 0xDEAD,
  big    = 0xFACE,
#if defined(_LIBCUDACXX_LITTLE_ENDIAN)
  native = little
#elif defined(_LIBCUDACXX_BIG_ENDIAN)
  native = big
#else
  native = 0xCAFE
#endif
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___BIT_ENDIAN_H
