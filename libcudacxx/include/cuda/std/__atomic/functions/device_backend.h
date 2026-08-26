//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ATOMIC_FUNCTIONS_DEVICE_BACKEND_H
#define _CUDA_STD___ATOMIC_FUNCTIONS_DEVICE_BACKEND_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CTK_AT_LEAST(13, 5) && _CCCL_HAS_NV_ATOMIC_BUILTINS()
#  include <cuda/std/__atomic/functions/cuda_nvvm.h>
#else
#  include <cuda/std/__atomic/functions/cuda_ptx.h>
#endif

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_CTK_AT_LEAST(13, 5) && _CCCL_HAS_NV_ATOMIC_BUILTINS()
using __cuda_atomic_device_backend = __cuda_atomic_nvvm_backend;
#else
using __cuda_atomic_device_backend = __cuda_atomic_ptx_backend;
#endif

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMIC_FUNCTIONS_DEVICE_BACKEND_H
