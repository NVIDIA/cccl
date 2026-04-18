//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___HIERARCHY_THREAD_LEVEL_H
#define _CUDA___HIERARCHY_THREAD_LEVEL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__fwd/hierarchy.h>
#  include <cuda/__hierarchy/native_hierarchy_level_base.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

struct _CCCL_DECLSPEC_EMPTY_BASES thread_level : __native_hierarchy_level_base<thread_level>
{
  using __product_type  = unsigned;
  using __allowed_above = __allowed_levels<block_level>;
  using __allowed_below = __allowed_levels<>;

  using __next_native_level = block_level;
};

_CCCL_GLOBAL_CONSTANT thread_level gpu_thread;

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___HIERARCHY_THREAD_LEVEL_H
