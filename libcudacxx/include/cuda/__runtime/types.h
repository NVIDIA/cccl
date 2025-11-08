//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___RUNTIME_TYPES_H
#define _CUDA___RUNTIME_TYPES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

using memory_location = ::cudaMemLocation;
#  if _CCCL_CTK_AT_LEAST(12, 2)
inline constexpr memory_location host_memory_location = {::cudaMemLocationTypeHost, 0};
#  endif // _CCCL_CTK_AT_LEAST(12, 2)

_CCCL_END_NAMESPACE_CUDA

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA___RUNTIME_TYPES_H
