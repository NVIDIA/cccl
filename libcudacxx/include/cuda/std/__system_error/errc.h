//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___SYSTEM_ERROR_ERRC_H
#define _LIBCUDACXX___SYSTEM_ERROR_ERRC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

enum class errc
{
  invalid_argument    = 22,
  result_out_of_range = 34,
#if _CCCL_OS(WINDOWS)
  value_too_large = 132,
#else // ^^^ _CCCL_OS(WINDOWS) ^^^ / vvv !_CCCL_OS(WINDOWS) vvv
  value_too_large = 75,
#endif // ^^^ !_CCCL_OS(WINDOWS) ^^^
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___SYSTEM_ERROR_ERRC_H
