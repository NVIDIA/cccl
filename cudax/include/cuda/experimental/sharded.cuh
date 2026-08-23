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
 */

#pragma once

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/shard.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>
