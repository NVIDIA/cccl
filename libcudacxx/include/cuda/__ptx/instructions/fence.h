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

#ifndef _CUDA_PTX_FENCE_H_
#define _CUDA_PTX_FENCE_H_

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

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

// 9.7.12.4. Parallel Synchronization and Communication Instructions: membar/fence
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence
#include <cuda/__ptx/instructions/generated/fence.h>
#include <cuda/__ptx/instructions/generated/fence_mbarrier_init.h>
#include <cuda/__ptx/instructions/generated/fence_proxy_alias.h>
#include <cuda/__ptx/instructions/generated/fence_proxy_async.h>
#include <cuda/__ptx/instructions/generated/fence_proxy_async_generic_sync_restrict.h>
#include <cuda/__ptx/instructions/generated/fence_proxy_tensormap_generic.h>
#include <cuda/__ptx/instructions/generated/fence_sync_restrict.h>

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_FENCE_H_
