//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cub/agent/single_pass_scan_operators.cuh>

#include "cccl/c/types.h"
#include <nvrtc/command_list.h>

struct scan_tile_state
{
  // scan_tile_state implements the same (host) interface as cub::ScanTileStateT, except
  // that it accepts the acummulator type as a runtime parameter rather than being
  // templated on it.
  //
  // Both specializations ScanTileStateT<T, true> and ScanTileStateT<T, false> - where the
  // bool parameter indicates whether `T` is primitive - are combined into a single type.

  void* d_tile_status; // d_tile_descriptors
  void* d_tile_partial;
  void* d_tile_inclusive;

  size_t description_bytes_per_tile;
  size_t payload_bytes_per_tile;

  scan_tile_state(size_t description_bytes_per_tile, size_t payload_bytes_per_tile)
      : d_tile_status(nullptr)
      , d_tile_partial(nullptr)
      , d_tile_inclusive(nullptr)
      , description_bytes_per_tile(description_bytes_per_tile)
      , payload_bytes_per_tile(payload_bytes_per_tile)
  {}

  cudaError_t Init(int num_tiles, void* d_temp_storage, size_t temp_storage_bytes)
  {
    void* allocations[3] = {};
    auto status          = cub::detail::tile_state_init(
      description_bytes_per_tile, payload_bytes_per_tile, num_tiles, d_temp_storage, temp_storage_bytes, allocations);
    if (status != cudaSuccess)
    {
      return status;
    }
    d_tile_status    = allocations[0];
    d_tile_partial   = allocations[1];
    d_tile_inclusive = allocations[2];
    return cudaSuccess;
  }

  cudaError_t AllocationSize(int num_tiles, size_t& temp_storage_bytes) const
  {
    return cub::detail::tile_state_allocation_size(
      temp_storage_bytes, description_bytes_per_tile, payload_bytes_per_tile, num_tiles);
  }
};

std::pair<size_t, size_t> get_tile_state_bytes_per_tile(
  cccl_type_info accum_t,
  const std::string& accum_cpp,
  const char** ptx_args,
  size_t num_ptx_args,
  const std::string& arch);
