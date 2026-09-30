//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Pulls `place_partition`-related names into `cuda::experimental::stf`.
 *
 * Include only after `stf::hash` is declared. Pulls in `place_partition.cuh`
 * (which depends on `async_resources_handle`); keep this separate from
 * `stf_places_extended_exports.cuh` to avoid an include cycle.
 */

#pragma once

#include <cuda/experimental/__places/place_partition.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh>

namespace cuda::experimental::stf
{
using ::cuda::experimental::places::make_grid;
using ::cuda::experimental::places::place_partition;
using ::cuda::experimental::places::place_partition_scope;
using ::cuda::experimental::places::place_partition_scope_to_string;
} // namespace cuda::experimental::stf
