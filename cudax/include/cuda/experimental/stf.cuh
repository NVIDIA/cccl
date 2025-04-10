//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 *
 * @brief Main include file for the CUDASTF library.
 */

#pragma once

#include <cuda/experimental/__stf/allocators/adapters.cuh>
#include <cuda/experimental/__stf/allocators/buddy_allocator.cuh>
#include <cuda/experimental/__stf/allocators/cached_allocator.cuh>
#include <cuda/experimental/__stf/allocators/pooled_allocator.cuh>
#include <cuda/experimental/__stf/allocators/uncached_allocator.cuh>
#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
// #include <cuda/experimental/__stf/internal/algorithm.cuh>
#include <cuda/experimental/__stf/internal/context.cuh>
// #include <cuda/experimental/__stf/internal/for_each_batched.cuh>
#include <cuda/experimental/__stf/internal/reducer.cuh>
#include <cuda/experimental/__stf/internal/scalar_interface.cuh>
#include <cuda/experimental/__stf/internal/task_dep.cuh>
#include <cuda/experimental/__stf/internal/void_interface.cuh>
#include <cuda/experimental/__stf/places/exec/cuda_stream.cuh>
#include <cuda/experimental/__stf/places/inner_shape.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>
#include <cuda/experimental/__stf/utility/run_once.cuh>
