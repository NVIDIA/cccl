// -*- C++ -*-
//===-------------------------- tgmath.h ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_TGMATH_H
#define _LIBCUDACXX_TGMATH_H

/*
    tgmath.h synopsis

#include <ctgmath>

*/

#include <__config>

#if defined(_CCCL_COMPILER_NVHPC) && defined(_CCCL_USE_IMPLICIT_SYSTEM_HEADER)
#pragma GCC system_header
#else // ^^^ _CCCL_COMPILER_NVHPC ^^^ / vvv !_CCCL_COMPILER_NVHPC vvv
_CCCL_IMPLICIT_SYSTEM_HEADER
#endif // !_CCCL_COMPILER_NVHPC

#ifdef __cplusplus

#include <ctgmath>

#else  // __cplusplus

#include_next <tgmath.h>

#endif  // __cplusplus

#endif  // _LIBCUDACXX_TGMATH_H
