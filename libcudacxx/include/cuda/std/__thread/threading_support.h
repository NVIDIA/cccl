// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___THREAD_THREADING_SUPPORT_H
#define _CUDA_STD___THREAD_THREADING_SUPPORT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__thread/sleep_for.h>
#include <cuda/std/__thread/yield.h>

#if _CCCL_HAS_THREAD_API(PTHREAD)
#  include <cuda/std/__thread/threading_support_pthread.h>
#elif _CCCL_HAS_THREAD_API(WIN32)
#  include <cuda/std/__thread/threading_support_win32.h>
#endif // _CCCL_HAS_THREAD_API(PTHREAD) || _CCCL_HAS_THREAD_API(WIN32)

#endif // _CUDA_STD___THREAD_THREADING_SUPPORT_H
