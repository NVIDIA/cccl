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
} // namespace

C2H_TEST("placement evaluation with a native mapper", "[places][placement]")
{
  stf_exec_place_handle grid = make_dev0_grid(2);

  const stf_dim4 dims{4 * MiB, 1, 1, 1};
  stf_placement_stats stats{};
  uint64_t bytes_per_pos[2] = {0, 0};

  int rc = stf_placement_evaluate(
    grid, stf_partition_fn_blocked(0), &dims, 1, /*probes=*/0, /*block_size=*/2 * MiB, &stats, bytes_per_pos);
  REQUIRE(rc == 0);

  REQUIRE(stats.total_bytes == 4 * MiB);
  REQUIRE(stats.vm_bytes == 4 * MiB);
  REQUIRE(stats.block_size == 2 * MiB);
  REQUIRE(stats.nblocks == 2);
  // Block-aligned blocked split over two positions: one allocation each and
  // every probe agrees with the block majority
  REQUIRE(stats.nallocs == 2);
  REQUIRE(stats.matching_samples == stats.total_samples);
  REQUIRE(bytes_per_pos[0] == 2 * MiB);
  REQUIRE(bytes_per_pos[1] == 2 * MiB);

  stf_exec_place_destroy(grid);
}

C2H_TEST("cute partition creation, accessors and leaf round trip", "[places][placement]")
{
  const stf_dim4 true_dims{10, 1, 1, 1};
  const stf_dim4 grid_dims{3, 1, 1, 1};
  const stf_partition_dim_spec spec[1] = {{STF_DIM_BLOCKED, 0, 0}};

  stf_cute_partition_handle part = stf_cute_partition_create(&true_dims, &grid_dims, spec, 1);
  REQUIRE(part != nullptr);

  stf_dim4 out{};
  stf_cute_partition_true_dims(part, &out);
  REQUIRE(out.x == 10);
  stf_cute_partition_padded_dims(part, &out);
  REQUIRE(out.x == 12); // ceil(10/3) * 3
  stf_cute_partition_grid_dims(part, &out);
  REQUIRE(out.x == 3);

  const size_t np = stf_cute_partition_num_place_leaves(part);
  const size_t nl = stf_cute_partition_num_local_leaves(part);
  REQUIRE(np == 1);
  REQUIRE(nl == 1);

  std::vector<uint64_t> p_ext(np), l_ext(nl);
  std::vector<int64_t> p_str(np), l_str(nl);
  std::vector<int> p_axes(np);
  stf_cute_partition_get_place_leaves(part, p_ext.data(), p_str.data(), p_axes.data());
  stf_cute_partition_get_local_leaves(part, l_ext.data(), l_str.data());
  REQUIRE(p_ext[0] == 3);
  REQUIRE(p_str[0] == 4); // chunk of 4 elements per place
  REQUIRE(p_axes[0] == 0);
  REQUIRE(l_ext[0] == 4);
  REQUIRE(l_str[0] == 1);
  REQUIRE(stf_cute_partition_place_offset(part, 1) == 4);

  // Rebuilding from the exported leaves must give an equivalent partition
  const stf_dim4 padded_dims{12, 1, 1, 1};
  stf_cute_partition_handle part2 = stf_cute_partition_from_leaves(
    p_ext.data(), p_str.data(), p_axes.data(), np, l_ext.data(), l_str.data(), nl, &padded_dims, &true_dims, &grid_dims);
  REQUIRE(part2 != nullptr);
  REQUIRE(stf_cute_partition_place_offset(part2, 2) == 8);

  // Leaves that do not tile the padded space exactly are rejected
  const uint64_t bad_ext[1] = {2};
  const int64_t bad_str[1]  = {1};
  const int bad_axes[1]     = {0};
  const stf_dim4 bad_grid{2, 1, 1, 1};
  stf_cute_partition_handle bad = stf_cute_partition_from_leaves(
    bad_ext, bad_str, bad_axes, 1, l_ext.data(), l_str.data(), nl, &padded_dims, &true_dims, &bad_grid);
  REQUIRE(bad == nullptr);

  stf_cute_partition_destroy(part2);
  stf_cute_partition_destroy(part);
  stf_cute_partition_destroy(nullptr); // must be a no-op
}

C2H_TEST("partition evaluation matches the equivalent native mapper", "[places][placement]")
{
  stf_exec_place_handle grid = make_dev0_grid(2);

  const stf_dim4 dims{8 * MiB, 1, 1, 1};
  const stf_partition_dim_spec spec[1] = {{STF_DIM_BLOCKED, 0, 0}};
  const stf_dim4 grid_dims{2, 1, 1, 1};

  stf_cute_partition_handle part = stf_cute_partition_create(&dims, &grid_dims, spec, 1);
  REQUIRE(part != nullptr);

  stf_placement_stats s_mapper{}, s_part{};
  uint64_t b_mapper[2] = {0, 0}, b_part[2] = {0, 0};

  REQUIRE(stf_placement_evaluate(grid, stf_partition_fn_blocked(0), &dims, 1, 0, 2 * MiB, &s_mapper, b_mapper) == 0);
  REQUIRE(stf_placement_evaluate_partition(grid, part, 1, 0, 2 * MiB, &s_part, b_part) == 0);

  REQUIRE(s_mapper.nblocks == s_part.nblocks);
  REQUIRE(s_mapper.nallocs == s_part.nallocs);
  REQUIRE(s_mapper.matching_samples == s_part.matching_samples);
  REQUIRE(b_mapper[0] == b_part[0]);
  REQUIRE(b_mapper[1] == b_part[1]);

  stf_cute_partition_destroy(part);
  stf_exec_place_destroy(grid);
}

C2H_TEST("shaped allocation on composite data places", "[places][placement][allocate]")
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
  std::vector<int> host(n, 42);
  REQUIRE(cudaMemcpy(ptr, host.data(), n * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
  std::vector<int> back(n, 0);
  REQUIRE(cudaMemcpy(back.data(), ptr, n * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(back[0] == 42);
  REQUIRE(back[n - 1] == 42);

  stf_data_place_deallocate(dp, ptr, n * sizeof(int), nullptr);
  stf_data_place_destroy(dp);

  // Same flow through a structured partition
  const stf_partition_dim_spec spec[1] = {{STF_DIM_BLOCKED, 0, 0}};
  const stf_dim4 grid_dims{2, 1, 1, 1};
  stf_cute_partition_handle part = stf_cute_partition_create(&dims, &grid_dims, spec, 1);
  REQUIRE(part != nullptr);
  stf_data_place_handle dpc = stf_data_place_composite_cute(grid, part);
  REQUIRE(dpc != nullptr);

  // Extents other than the partition's true extents are rejected
  const stf_dim4 other_dims{n / 2, 1, 1, 1};
  REQUIRE(stf_data_place_allocate_nd(dpc, &other_dims, sizeof(int), nullptr) == nullptr);

  void* ptr2 = stf_data_place_allocate_nd(dpc, &dims, sizeof(int), nullptr);
  REQUIRE(ptr2 != nullptr);
  REQUIRE(cudaMemcpy(ptr2, host.data(), n * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
  stf_data_place_deallocate(dpc, ptr2, n * sizeof(int), nullptr);

  stf_data_place_destroy(dpc);
  stf_cute_partition_destroy(part);
  stf_exec_place_destroy(grid);
}
