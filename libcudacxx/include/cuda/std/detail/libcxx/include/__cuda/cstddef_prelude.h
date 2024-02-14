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

#ifndef __cuda_std__
#error "<__cuda/cstddef_prelude> should only be included in from <cuda/std/cstddef>"
#endif // __cuda_std__

#ifndef _CCCL_COMPILER_NVRTC
#include <cstddef>
#include <stddef.h>
#else
#define offsetof(type, member) (_CUDA_VSTD::size_t)((char*)&(((type *)0)->member) - (char*)0)
#endif // _CCCL_COMPILER_NVRTC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

typedef decltype(nullptr) nullptr_t;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CUDA_CSTDDEF_PRELUDE_H
