//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___INTERNAL_PSTL_CONFIG_H
#define _CUDA_STD___INTERNAL_PSTL_CONFIG_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

#define _CCCL_HAS_BACKEND_CUDA() _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)
#define _CCCL_HAS_BACKEND_OMP()  0
#define _CCCL_HAS_BACKEND_TBB()  0

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___INTERNAL_PSTL_CONFIG_H
