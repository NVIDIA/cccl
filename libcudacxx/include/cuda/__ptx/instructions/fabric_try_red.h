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

#ifndef _CUDA_PTX_FABRIC_TRY_RED_H_
#define _CUDA_PTX_FABRIC_TRY_RED_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__ptx/ptx_dot_variants.h>
#include <cuda/__ptx/ptx_helper_functions.h>
#include <cuda/std/cstdint>

#include <nv/target> // __CUDA_MINIMUM_ARCH__ and friends

#include <cuda/std/__cccl/prologue.h>

// Forward-declare __half and __nv_bfloat16. The cuda_fp16.h and cuda_bf16.h are
// expensive to include. The APIs use only pointers, so we do not have to define
// the types. If the user wants to use these types, it is their responsibility
// to include the headers.
#if _LIBCUDACXX_HAS_NVFP16()
struct __half;
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
struct __nv_bfloat16;
#endif // _LIBCUDACXX_HAS_NVBF16()

_CCCL_BEGIN_NAMESPACE_CUDA_PTX

#include <cuda/__ptx/instructions/generated/fabric_try_red.h>

_CCCL_END_NAMESPACE_CUDA_PTX

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_PTX_FABRIC_TRY_RED_H_
