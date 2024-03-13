// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___NEW_ALIGN_VAL_H
#define _LIBCUDACXX___NEW_ALIGN_VAL_H

#ifndef __cuda_std__
#  include <__config>
#endif //__cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_STD_VER >= 2017

// The standard explicitly requires the use of `std::align_val_t` so we need to use an alias
#if !defined(_CCCL_COMPILER_NVRTC)
#include <new>
#endif // !_CCCL_COMPILER_NVRTC

_LIBCUDACXX_BEGIN_NAMESPACE_STD_NOVERSION // purposefully not using versioning namespace

using ::std::align_val_t;

_LIBCUDACXX_END_NAMESPACE_STD_NOVERSION

#endif // _CCCL_STD_VER >= 2017

#endif // _LIBCUDACXX___NEW_ALIGN_VAL_H
