// -*- C++ -*-
//===--------------------------- complex.h --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_COMPLEX_H
#define _LIBCUDACXX_COMPLEX_H

/*
    complex.h synopsis

#include <ccomplex>

*/

#include <__config>

#if defined(_CCCL_COMPILER_NVHPC) && defined(_CCCL_USE_IMPLICIT_SYSTEM_DEADER)
#pragma GCC system_header
#else // ^^^ _CCCL_COMPILER_NVHPC ^^^ / vvv !_CCCL_COMPILER_NVHPC vvv
_CCCL_IMPLICIT_SYSTEM_HEADER
#endif // !_CCCL_COMPILER_NVHPC

#ifdef __cplusplus

#include <ccomplex>

#else  // __cplusplus

#include_next <complex.h>

#endif  // __cplusplus

#endif  // _LIBCUDACXX_COMPLEX_H
