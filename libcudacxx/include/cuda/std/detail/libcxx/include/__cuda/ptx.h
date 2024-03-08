// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_PTX_H
#define  _LIBCUDACXX___CUDA_PTX_H

#ifndef __cuda_std__
#error "<__cuda/ptx.h> should only be included in from <cuda/ptx>"
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
 * The cuda::ptx namespace intends to provide PTX wrappers for new hardware
 * features and new PTX instructions so that they can be experimented with
 * before higher-level C++ APIs are designed and developed.
 *
 * The wrappers have the following responsibilities:
 *
 * - They must prevent any PTX assembler errors, that is:
 *   - They are defined only for versions of the CUDA Toolkit in which nvcc/ptxas
 *     actually recognizes the instruction.
 *   - Sizes and types of parameters are correct.
 * - They must convert state spaces correctly.
 * - They adhere to the libcu++ coding standards of using:
 *   - Reserved identifiers for all parameters, variables. E.g. `__meow` or `_Woof`
 *   - _CUDA_VSTD:: namespace for types
 *
 * The wrappers should not do the following:
 *
 * - Use any non-native types. For example, an mbarrier instruction wrapper
 *   takes the barrier address as a uint64_t pointer.
 *
 * This header is intended for:
 *
 * - internal consumption by higher-level APIs such as cuda::barrier,
 * - outside developers who want to experiment with the latest features of the
 *   hardware.
 *
 * Stability:
 *
 * - These headers are intended to present a stable API (not ABI) within one
 *   major version of the CTK. This means that:
 *   - All functions are marked inline
 *   - The type of a function parameter can be changed to be more generic if
 *     that means that code that called the original version can still be
 *     compiled.
 *
 * - Good exposure of the PTX should be high priority. If, at a new major
 *   version, we face a difficult choice between breaking backward-compatibility
 *   and an improvement of the PTX exposure, we will tend to the latter option
 *   more easily than in other parts of libcu++.
 *
 * Code organization:
 *
 * - Each instruction is in a separate file and is included below.
 * - Some helper function and types can be found in ptx/ptx_helper_functions.h and ptx/ptx_dot_variants.h.
 */

#include "ptx/instructions/barrier_cluster.h"
#include "ptx/instructions/cp_async_bulk.h"
#include "ptx/instructions/cp_async_bulk_commit_group.h"
#include "ptx/instructions/cp_async_bulk_tensor.h"
#include "ptx/instructions/cp_async_bulk_wait_group.h"
#include "ptx/instructions/cp_reduce_async_bulk.h"
#include "ptx/instructions/cp_reduce_async_bulk_tensor.h"
#include "ptx/instructions/fence.h"
#include "ptx/instructions/get_sreg.h"
#include "ptx/instructions/getctarank.h"
#include "ptx/instructions/mbarrier_arrive.h"
#include "ptx/instructions/mbarrier_init.h"
#include "ptx/instructions/mbarrier_wait.h"
#include "ptx/instructions/red_async.h"
#include "ptx/instructions/st_async.h"
#include "ptx/instructions/tensormap_cp_fenceproxy.h"
#include "ptx/instructions/tensormap_replace.h"

#endif // _LIBCUDACXX___CUDA_PTX_H
