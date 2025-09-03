//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_TARGET_MACROS_H
#define __CCCL_TARGET_MACROS_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <nv/target>

#if _CCCL_CUDA_COMPILATION()
#if _CCCL_CUDA_COMPILER(NVHPC)
#  define _CCCL_TARGET_IS_DEVICE     target (nv::target::is_device)
#  define _CCCL_TARGET_IS_HOST       target (nv::target::is_host)
#  define _CCCL_TARGET_ANY           target (nv::target::any_target)
#  define _CCCL_TARGET_NONE          target (nv::target::no_target)
#  define _CCCL_TARGET_IS_EXACTLY(X) target (nv::target::is_exactly(sm_##X))
#  define _CCCL_TARGET_PROVIDES(X)   target (nv::target::provides(sm_##X))
#else // ^^^ _CCCL_CUDA_COMPILER(NVHPC) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVHPC) vvv
#  define _CCCL_TARGET_IS_DEVICE     constexpr (_CCCL_PTX_ARCH() != 0)
#  define _CCCL_TARGET_IS_HOST       constexpr (_CCCL_PTX_ARCH() == 0)
#  define _CCCL_TARGET_ANY           constexpr (true)
#  define _CCCL_TARGET_NONE          constexpr (false)
#  define _CCCL_TARGET_IS_EXACTLY(X) constexpr (_CCCL_PTX_ARCH() == (X * 10))
#  define _CCCL_TARGET_PROVIDES(X)   constexpr (_CCCL_PTX_ARCH() >= (X * 10))
#endif // ^^^ !_CCCL_CUDA_COMPILER(NVHPC) ^^^
#else // ^^^ _CCCL_CUDA_COMPILATION() ^^^ / vvv !_CCCL_CUDA_COMPILATION() vvv
#  define _CCCL_TARGET_IS_DEVICE     constexpr (false)
#  define _CCCL_TARGET_IS_HOST       constexpr (true)
#  define _CCCL_TARGET_ANY           constexpr (true)
#  define _CCCL_TARGET_NONE          constexpr (false)
#  define _CCCL_TARGET_IS_EXACTLY(X) constexpr (false)
#  define _CCCL_TARGET_PROVIDES(X)   constexpr (false)
#endif // ^^^ !_CCCL_CUDA_COMPILATION() ^^^

#endif // __CCCL_TARGET_MACROS_H
