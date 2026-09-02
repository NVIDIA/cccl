//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief cute_partition_descriptor::try_block_owners — the analytic page
 * plan — cross-validated against a brute-force per-element owner scan, and
 * exercised end-to-end through a localized_array allocation.
 */

#include <cuda/experimental/__stf/localization/composite_slice.cuh>
#include <cuda/experimental/__stf/utility/unittest.cuh>

#include <map>

using namespace cuda::experimental::stf;
using namespace cuda::experimental::places;

namespace
{
// Brute force: owner of every byte via per-element owner() queries.
::std::vector<pos4> brute_block_owners(
  const cute_partition_descriptor& part,
  size_t block_size_bytes,
  size_t elemsize,
  size_t total_elems,
  dim4 data_dims,
  size_t* misplaced_bytes)
{
  const size_t total_bytes = total_elems * elemsize;
  const size_t nblocks     = (total_bytes + block_size_bytes - 1) / block_size_bytes;
  *misplaced_bytes         = 0;
  ::std::vector<pos4> owners;
  for (size_t b = 0; b < nblocks; b++)
  {
    ::std::map<::std::array<ssize_t, 4>, size_t> census;
    const size_t lo = b * block_size_bytes;
    const size_t hi = ::std::min((b + 1) * block_size_bytes, total_bytes);
    for (size_t byte = lo; byte < hi; byte++)
    {
      const pos4 o = part.owner(data_dims.index_to_pos(byte / elemsize));
      census[{o.x, o.y, o.z, o.t}]++;
    }
    size_t best = 0;
    ::std::array<ssize_t, 4> bp{};
    for (const auto& e : census)
    {
      if (e.second > best)
      {
        best = e.second;
        bp   = e.first;
      }
    }
    owners.push_back(pos4(bp[0], bp[1], bp[2], bp[3]));
    *misplaced_bytes += (hi - lo) - best;
  }
  return owners;
}

void check_case(dim4 data_dims, const ::std::vector<dim_spec>& spec, dim4 grid_dims, size_t elemsize, size_t block)
{
  const auto part          = make_partition_descriptor(data_dims, spec, grid_dims);
  const size_t total_elems = data_dims.size();

  size_t misplaced = 0;
  auto plan        = part.try_block_owners(block, elemsize, &misplaced);
  if (!plan)
  {
    return; // dense: sampled tier takes over (covered elsewhere)
  }

  size_t brute_misplaced = 0;
  const auto brute       = brute_block_owners(part, block, elemsize, total_elems, data_dims, &brute_misplaced);
  EXPECT(plan->size() == brute.size());
  EXPECT(misplaced == brute_misplaced);

  // The runs path must agree with the brute-force census wherever the
  // strict quotient exists: zero misplacement, and every block of every run
  // owned by the brute owner. Where it declines, the census must have found
  // at least one straddled (misplaced) block or an in-block boundary.
  if (const auto runs = part.try_block_runs(block, elemsize))
  {
    EXPECT(misplaced == 0);
    size_t covered = 0;
    for (const auto& r : *runs)
    {
      // strict tiling: each run starts where the previous ended and stays
      // in bounds (covered == total alone would accept overlap/gap pairs)
      EXPECT(r.first_block == covered);
      EXPECT(r.num_blocks > 0);
      EXPECT(r.num_blocks <= brute.size() - covered);
      for (size_t b = r.first_block; b < r.first_block + r.num_blocks; b++)
      {
        EXPECT(brute[b] == r.owner);
      }
      covered += r.num_blocks;
    }
    EXPECT(covered == brute.size());
  }
  for (size_t b = 0; b < brute.size(); b++)
  {
    // majority may tie: accept the analytic owner iff its byte count ties the
    // brute majority; equality of misplaced bytes above already pins that.
    if (!((*plan)[b] == brute[b]))
    {
      // re-census this block for the analytic owner's count
      size_t analytic_bytes = 0, brute_bytes = 0;
      const size_t lo = b * block;
      const size_t hi = ::std::min((b + 1) * block, total_elems * elemsize);
      for (size_t byte = lo; byte < hi; byte++)
      {
        const pos4 o = part.owner(data_dims.index_to_pos(byte / elemsize));
        analytic_bytes += (o == (*plan)[b]);
        brute_bytes += (o == brute[b]);
      }
      EXPECT(analytic_bytes == brute_bytes); // a genuine tie
    }
  }
}

void property_suite()
{
  const ::std::vector<size_t> grids = {1, 2, 3, 4, 6, 8};
  for (size_t g : grids)
  {
    for (size_t elemsize : {2, 4})
    {
      for (size_t block : {16, 64, 256})
      {
        // 1-D blocked / cyclic / block_cyclic
        check_case(dim4(13), {{dim_policy::blocked, 0, 0}}, dim4(g), elemsize, block);
        check_case(dim4(48), {{dim_policy::blocked, 0, 0}}, dim4(g), elemsize, block);
        check_case(dim4(64), {{dim_policy::cyclic, 0, 0}}, dim4(g), elemsize, block);
        check_case(dim4(64), {{dim_policy::block_cyclic, 0, 4}}, dim4(g), elemsize, block);
        // 2-D: outer blocked, inner whole (expert-major shape)
        check_case(dim4(12, 16), {{dim_policy::blocked, 0, 0}, {}}, dim4(g), elemsize, block);
        // 2-D: inner cyclic
        check_case(dim4(12, 16), {{}, {dim_policy::cyclic, 0, 0}}, dim4(g), elemsize, block);
      }
    }
  }
  // tiled 2-D on a (2,2) grid + 3-D with a middle whole dim
  for (size_t elemsize : {2, 4})
  {
    for (size_t block : {16, 64, 256})
    {
      check_case(dim4(12, 16), {{dim_policy::blocked, 0, 0}, {dim_policy::blocked, 1, 0}}, dim4(2, 2), elemsize, block);
      check_case(
        dim4(6, 8, 4), {{dim_policy::blocked, 0, 0}, {}, {dim_policy::blocked, 1, 0}}, dim4(2, 2), elemsize, block);
    }
  }
  // dense detection: element-cyclic far below the block size must decline
  const auto dense = make_partition_descriptor(dim4(1 << 22), {{dim_policy::cyclic, 0, 0}}, dim4(2));
  size_t mis       = 0;
  EXPECT(!dense.try_block_owners(2 * 1024 * 1024, 4, &mis).has_value());
}

void end_to_end_allocation()
{
  // Exercise the provider path through a real VMM allocation: a 2-place
  // grid on the current device (placement plumbing, merge, stats).
  const auto d0 = exec_place::device(cuda_try<cudaGetDevice>());
  ::std::vector<exec_place> places{d0, d0};
  const auto grid = make_grid(mv(places));

  const size_t n = (6 * 1024 * 1024) + 512; // forces one straddle block
  const dim4 data_dims(n);
  const auto part = make_partition_descriptor(data_dims, {{dim_policy::blocked, 0, 0}}, grid.get_dims());

  localized_array arr(
    grid, make_partition_placement_provider(part, data_dims, data_dims.size(), sizeof(int)), n, sizeof(int), data_dims);
  const auto& st = arr.get_stats();
  // exact plan: sample counters hold byte counts
  EXPECT(st.total_samples == n * sizeof(int));
  size_t mis        = 0;
  const auto owners = part.try_block_owners(st.block_size, sizeof(int), &mis);
  EXPECT(owners.has_value() == true);
  EXPECT(st.matching_samples == st.total_samples - mis);
  EXPECT(st.nallocs <= 3); // two shards + at most one straddle merge break
}
void malformed_providers_throw()
{
  const auto d0 = exec_place::device(cuda_try<cudaGetDevice>());
  ::std::vector<exec_place> places{d0, d0};
  const auto grid = make_grid(mv(places));
  const size_t n  = 4 * 1024 * 1024; // 16 MB of int: 8 blocks at 2 MB
  const dim4 dims(n);

  auto expect_throw = [&](auto&& provider) {
    bool thrown = false;
    try
    {
      localized_array arr(grid, provider, n, sizeof(int), dims);
    }
    catch (const ::std::invalid_argument&)
    {
      thrown = true;
    }
    EXPECT(thrown);
  };
  // gap: second run skips a block
  expect_throw([](size_t, size_t nblocks, localized_stats&) {
    return ::std::vector<block_run>{{pos4(0), 0, 1}, {pos4(1), 2, nblocks - 2}};
  });
  // short: does not cover the final block
  expect_throw([](size_t, size_t nblocks, localized_stats&) {
    return ::std::vector<block_run>{{pos4(0), 0, nblocks - 1}};
  });
  // zero-length run
  expect_throw([](size_t, size_t nblocks, localized_stats&) {
    return ::std::vector<block_run>{{pos4(0), 0, 0}, {pos4(1), 0, nblocks}};
  });
}
} // namespace

int main()
{
  property_suite();
  end_to_end_allocation();
  malformed_providers_throw();
  printf("cute_block_owners: all checks passed\n");
  return 0;
}
