//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 *
 * @brief Main include file for the sharded containers and algorithms.
 *
 * Sharded arrays partition one logical array across places (devices or
 * sub-device locality domains) inside a single process — the rung of the
 * cooperation-scope ladder where a common address space is still shared while
 * each byte has exactly one physical home. Algorithms follow the ladder's
 * recipe: run the device-scope primitive per place, combine through what the
 * rung shares.
 *
 * Built on the standalone places layer (`cuda/experimental/places.cuh`), in
 * particular `place_group` (execution resources) and `localized_array` (the
 * VMM backing of `sharded_array<T>::allocate_contiguous`).
 *
 * Layout of `__sharded/`: `concepts/` (what the algorithms are written
 * against), `container/` (the reference models), `composition/` (the
 * cross-lane verbs), `algorithm/<name>/` (the verbs), `engine/` (the drivers
 * the algorithms include themselves — not part of this umbrella), `sparse/`
 * and `algorithm/random/` (opt-in vendor tiers, not part of this umbrella)
 * and `reference/` (non-shipping comparison implementations, never
 * included).
 */

#pragma once

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/algorithm/adjacent_difference/adjacent_difference.cuh>
#include <cuda/experimental/__sharded/algorithm/copy_if/copy_if.cuh>
#include <cuda/experimental/__sharded/algorithm/count/count.cuh>
#include <cuda/experimental/__sharded/algorithm/fill/fill.cuh>
#include <cuda/experimental/__sharded/algorithm/for_each_shard/for_each_shard.cuh>
#include <cuda/experimental/__sharded/algorithm/histogram/histogram.cuh>
#include <cuda/experimental/__sharded/algorithm/reduce/reduce.cuh>
#include <cuda/experimental/__sharded/algorithm/scan/scan.cuh>
#include <cuda/experimental/__sharded/algorithm/segmented_reduce/segmented_reduce.cuh>
#include <cuda/experimental/__sharded/algorithm/sort/sort.cuh>
#include <cuda/experimental/__sharded/algorithm/transform/transform.cuh>
#include <cuda/experimental/__sharded/algorithm/unique/unique.cuh>
#include <cuda/experimental/__sharded/composition/fork_join.cuh>
#include <cuda/experimental/__sharded/composition/pinned_staging.cuh>
#include <cuda/experimental/__sharded/composition/verbs.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/container/csr.cuh>
#include <cuda/experimental/__sharded/container/default_envs.cuh>
#include <cuda/experimental/__sharded/container/shard.cuh>
#include <cuda/experimental/__sharded/container/sharded_array.cuh>
