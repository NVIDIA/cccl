//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___INTERNAL_DLPACK_H
#define _CUDA___INTERNAL_DLPACK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_DLPACK()

#  include <dlpack/dlpack.h>

#  define _CCCL_DLPACK_AT_LEAST(_MAJOR, _MINOR) \
    (DLPACK_MAJOR_VERSION > (_MAJOR) || (DLPACK_MAJOR_VERSION == (_MAJOR) && DLPACK_VERSION_MINOR >= (_MINOR)))
#  define _CCCL_DLPACK_BELOW(_MAJOR, _MINOR) (!_CCCL_DLPACK_AT_LEAST(_MAJOR, _MINOR))

#  if DLPACK_MAJOR_VERSION != 1
#    error "Unsupported DLPack version, only version 1 is currently supported"
#  endif // DLPACK_MAJOR_VERSION != 1

#endif // _CCCL_HAS_DLPACK()

#endif // _CUDA___INTERNAL_DLPACK_H
