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
 *
 * STF pulls `cuda::experimental::places` into `cuda::experimental::stf` via
 * `__stf/internal/stf_places_into_stf_core.cuh` (after `places.cuh` in STF-only
 * headers), `stf_places_extended_exports.cuh` (after `stf::hash`; avoids including
 * `place_partition.cuh`), and `stf_places_partition_into_stf.cuh` where partition
 * APIs are needed. Places headers do not include STF for that bridge.
 */

#pragma once

#include <cuda/experimental/__stf/allocators/adapters.cuh>
#include <cuda/experimental/__stf/allocators/buddy_allocator.cuh>
#include <cuda/experimental/__stf/allocators/cached_allocator.cuh>
#include <cuda/experimental/__stf/allocators/pooled_allocator.cuh>
#include <cuda/experimental/__stf/allocators/uncached_allocator.cuh>
#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
// #include <cuda/experimental/__stf/internal/algorithm.cuh>
#include <cuda/experimental/__places/exec/cuda_stream.cuh>
#include <cuda/experimental/__places/exec/green_context.cuh>
#include <cuda/experimental/__stf/internal/context.cuh>
#include <cuda/experimental/__stf/internal/inner_shape.cuh>
#include <cuda/experimental/__stf/internal/reducer.cuh>
#include <cuda/experimental/__stf/internal/scalar_interface.cuh>
#include <cuda/experimental/__stf/internal/task_dep.cuh>
#include <cuda/experimental/__stf/internal/void_interface.cuh>
#include <cuda/experimental/__stf/stackable/stackable_ctx.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>
#include <cuda/experimental/__stf/utility/run_once.cuh>
