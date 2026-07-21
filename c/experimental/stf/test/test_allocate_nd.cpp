//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <vector>

#include <cuda_runtime.h>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/experimental/stf/stf.h>

namespace
{
constexpr uint64_t MiB = 1024 * 1024;

stf_exec_place_handle make_dev0_grid(size_t nplaces)
{
  std::vector<stf_exec_place_handle> places(nplaces);
  for (auto& place : places)
  {
    place = stf_exec_place_device(0);
    REQUIRE(place != nullptr);
  }
  stf_exec_place_handle grid = stf_exec_place_grid_create(places.data(), nplaces, nullptr);
  REQUIRE(grid != nullptr);
  for (auto& place : places)
  {
    stf_exec_place_destroy(place);
  }
  return grid;
}

void check_device_round_trip(void* ptr, uint64_t n)
{
  std::vector<int> host(n, 42);
  REQUIRE(cudaMemcpy(ptr, host.data(), n * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
  std::vector<int> back(n, 0);
  REQUIRE(cudaMemcpy(back.data(), ptr, n * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(back[0] == 42);
  REQUIRE(back[n - 1] == 42);
}
} // namespace

C2H_TEST("shaped allocation on an ordinary data place", "[places][allocate]")
{
  const uint64_t n = MiB; // ints
  const stf_dim4 dims{n, 1, 1, 1};

  stf_data_place_handle dp = stf_data_place_device(0);
  REQUIRE(dp != nullptr);

  // On a non-composite place the geometry degenerates to a byte count
  void* ptr = stf_data_place_allocate_nd(dp, &dims, sizeof(int), nullptr);
  REQUIRE(ptr != nullptr);
  check_device_round_trip(ptr, n);

  stf_data_place_deallocate(dp, ptr, n * sizeof(int), nullptr);
  stf_data_place_destroy(dp);
}

C2H_TEST("shaped allocation on composite data places", "[places][allocate]")
{
  stf_exec_place_handle grid = make_dev0_grid(2);

  const uint64_t n = MiB; // ints
  const stf_dim4 dims{n, 1, 1, 1};

  stf_data_place_handle dp = stf_data_place_composite(grid, stf_partition_fn_blocked(0));
  REQUIRE(dp != nullptr);

  // A byte count alone cannot carry the tensor geometry: must fail cleanly
  void* bad = stf_data_place_allocate(dp, static_cast<ptrdiff_t>(n * sizeof(int)), nullptr);
  REQUIRE(bad == nullptr);

  void* ptr = stf_data_place_allocate_nd(dp, &dims, sizeof(int), nullptr);
  REQUIRE(ptr != nullptr);

  // Memory must be usable from the device
  check_device_round_trip(ptr, n);

  stf_data_place_deallocate(dp, ptr, n * sizeof(int), nullptr);
  stf_data_place_destroy(dp);

  // Same flow through the native cyclic partition function
  stf_data_place_handle dpc = stf_data_place_composite(grid, stf_partition_fn_cyclic());
  REQUIRE(dpc != nullptr);

  void* ptr2 = stf_data_place_allocate_nd(dpc, &dims, sizeof(int), nullptr);
  REQUIRE(ptr2 != nullptr);
  check_device_round_trip(ptr2, n);

  stf_data_place_deallocate(dpc, ptr2, n * sizeof(int), nullptr);
  stf_data_place_destroy(dpc);
  stf_exec_place_destroy(grid);
}
