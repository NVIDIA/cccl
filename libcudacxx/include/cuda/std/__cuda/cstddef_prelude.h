// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_CSTDDEF_PRELUDE_H
#define _LIBCUDACXX___CUDA_CSTDDEF_PRELUDE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#ifndef _CCCL_COMPILER_NVRTC
#  include <cstddef>

#  include <stddef.h>
#else
#  define offsetof(type, member) (_CUDA_VSTD::size_t)((char*) &(((type*) 0)->member) - (char*) 0)
#endif // _CCCL_COMPILER_NVRTC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

typedef decltype(nullptr) nullptr_t;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CUDA_CSTDDEF_PRELUDE_H
