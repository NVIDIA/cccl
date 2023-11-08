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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#ifdef __cplusplus

#include <ctgmath>

#else  // __cplusplus

#include_next <tgmath.h>

#endif  // __cplusplus

#endif  // _LIBCUDACXX_TGMATH_H
