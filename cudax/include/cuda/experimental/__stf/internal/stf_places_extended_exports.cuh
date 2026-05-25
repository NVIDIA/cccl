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
 * @brief Additional `places` names and `stf::hash` specializations for place keys.
 *
 * Includes `hash.cuh` first so this header stays valid under clang-format
 * `SortIncludes` (all `<cuda/experimental/...>` lines are reordered lexically).
 *
 * Do not duplicate names from `stf_places_into_stf_core.cuh`.
 *
 * Does not include `place_partition.cuh` (it depends on `async_resources_handle`);
 * use `stf_places_partition_into_stf.cuh` where `place_partition` / `make_grid`
 * names are needed in `stf`.
 */

#pragma once

#include <cuda/experimental/__places/localized_array.cuh>
#include <cuda/experimental/__places/machine.cuh>
#include <cuda/experimental/__places/partitions/blocked_partition.cuh>
#include <cuda/experimental/__places/partitions/tiled_partition.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__places/stream_pool.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh>
#if _CCCL_CTK_AT_LEAST(12, 4)
#  include <cuda/experimental/__places/exec/green_context.cuh>
#endif // _CCCL_CTK_AT_LEAST(12, 4)

namespace cuda::experimental::stf
{
using ::cuda::experimental::places::blocked_partition;
using ::cuda::experimental::places::cyclic_partition;
#if _CCCL_CTK_AT_LEAST(12, 4)
using ::cuda::experimental::places::green_context_helper;
using ::cuda::experimental::places::green_ctx_view;
#endif // _CCCL_CTK_AT_LEAST(12, 4)
using ::cuda::experimental::places::get_device_from_stream;
using ::cuda::experimental::places::k_no_stream_id;
using ::cuda::experimental::places::localized_array;
using ::cuda::experimental::places::partition_fn_t;
using ::cuda::experimental::places::stream_pool;
using ::cuda::experimental::places::tiled;
using ::cuda::experimental::places::tiled_partition;
using ::cuda::experimental::places::reserved::machine;

template <>
struct hash<::cuda::experimental::places::exec_place>
    : ::cuda::experimental::places::hash<::cuda::experimental::places::exec_place>
{};

template <>
struct hash<::cuda::experimental::places::data_place>
    : ::cuda::experimental::places::hash<::cuda::experimental::places::data_place>
{};
} // namespace cuda::experimental::stf
