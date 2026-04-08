//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 *
 * @brief Main include file for the standalone CUDA places library.
 *
 * Provides data_place, exec_place, stream_pool, green context support,
 * place partitioning utilities, and partition strategies (`cyclic_shape`,
 * `cyclic_partition`, `blocked_partition`, `tiled_partition`) without pulling
 * in the full CUDASTF task-graph framework.
 */

#pragma once

#include <cuda/experimental/__places/exec/cuda_stream.cuh>
#include <cuda/experimental/__places/exec/green_context.cuh>
#include <cuda/experimental/__places/exec/green_ctx_view.cuh>
#include <cuda/experimental/__places/machine.cuh>
#include <cuda/experimental/__places/partitions/blocked_partition.cuh>
#include <cuda/experimental/__places/partitions/cyclic_shape.cuh>
#include <cuda/experimental/__places/partitions/tiled_partition.cuh>
#include <cuda/experimental/__places/place_partition.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__places/stream_pool.cuh>
