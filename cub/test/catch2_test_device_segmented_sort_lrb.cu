// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_segmented_sort_lrb.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "cub_test_macros.h"
#include <c2h/vector.h>

using cub::detail::segmented_sort_lrb::lrb_policy;
using cub::detail::segmented_sort_lrb::lrb_plan;
using cub::detail::segmented_sort_lrb::lrb_shape;
using cub::detail::segmented_sort_lrb::lrb_summary;
using cub::detail::segmented_sort_lrb::lrb_tier;
using cub::detail::segmented_sort_lrb::map_segment_length;
using cub::detail::segmented_sort_lrb::DispatchSegmentedSortLrb;

using count_t  = std::uint32_t;
using offset_t = std::int32_t;
using policy_t = lrb_policy<>;

struct host_plan
{
  lrb_summary<count_t> summary{};
  std::vector<count_t> segment_ids;
  std::vector<count_t> warp_offsets;
  std::vector<count_t> small_offsets;
  std::vector<count_t> large_offsets;
  std::vector<count_t> overflow_offsets;
  std::vector<count_t> warp_groups;
  std::vector<count_t> small_groups;
  std::vector<count_t> large_groups;
};

template <typename OffsetT>
lrb_shape expected_shape(OffsetT length)
{
  return map_segment_length<policy_t>(length);
}

std::vector<offset_t> exclusive_csr(const std::vector<offset_t>& degrees)
{
  std::vector<offset_t> offsets(degrees.size() + 1, 0);
  std::partial_sum(degrees.begin(), degrees.end(), offsets.begin() + 1);
  return offsets;
}

host_plan run_planner(const std::vector<offset_t>& begin, const std::vector<offset_t>& end)
{
  REQUIRE(begin.size() == end.size());
  const int num_segments = static_cast<int>(begin.size());

  c2h::device_vector<offset_t> d_begin = begin;
  c2h::device_vector<offset_t> d_end   = end;

  using dispatch_t = DispatchSegmentedSortLrb<const offset_t*, const offset_t*, count_t, policy_t>;

  lrb_plan<count_t> plan{};
  std::size_t bytes = 0;
  REQUIRE_CUDART(dispatch_t::Dispatch(nullptr, bytes, nullptr, nullptr, num_segments, plan));
  REQUIRE(bytes > 0);

  c2h::device_vector<std::uint8_t> temp(bytes);
  REQUIRE_CUDART(dispatch_t::Dispatch(
    thrust::raw_pointer_cast(temp.data()),
    bytes,
    thrust::raw_pointer_cast(d_begin.data()),
    thrust::raw_pointer_cast(d_end.data()),
    num_segments,
    plan));

  host_plan host{};
  REQUIRE_CUDART(cudaMemcpy(&host.summary, plan.d_summary, sizeof(host.summary), cudaMemcpyDeviceToHost));

  host.segment_ids.resize(host.summary.num_assigned);
  if (!host.segment_ids.empty())
  {
    REQUIRE_CUDART(cudaMemcpy(
      host.segment_ids.data(),
      plan.d_segment_ids,
      host.segment_ids.size() * sizeof(count_t),
      cudaMemcpyDeviceToHost));
  }

  auto copy_offsets = [](count_t* ptr, int n) {
    std::vector<count_t> out(static_cast<std::size_t>(n));
    REQUIRE_CUDART(cudaMemcpy(out.data(), ptr, out.size() * sizeof(count_t), cudaMemcpyDeviceToHost));
    return out;
  };

  host.warp_offsets     = copy_offsets(plan.d_warp_bin_offsets, policy_t::warp_bin_count + 1);
  host.small_offsets    = copy_offsets(plan.d_small_block_bin_offsets, policy_t::small_block_bin_count + 1);
  host.large_offsets    = copy_offsets(plan.d_large_block_bin_offsets, policy_t::large_block_bin_count + 1);
  host.overflow_offsets = copy_offsets(plan.d_overflow_offsets, policy_t::overflow_bin_count + 1);
  host.warp_groups      = copy_offsets(plan.d_warp_group_offsets, policy_t::warp_bin_count + 1);
  host.small_groups     = copy_offsets(plan.d_small_block_group_offsets, policy_t::small_block_bin_count + 1);
  host.large_groups     = copy_offsets(plan.d_large_block_group_offsets, policy_t::large_block_bin_count + 1);
  return host;
}

host_plan run_planner_from_degrees(const std::vector<offset_t>& degrees)
{
  const auto offsets = exclusive_csr(degrees);
  return run_planner(std::vector<offset_t>(offsets.begin(), offsets.end() - 1),
                     std::vector<offset_t>(offsets.begin() + 1, offsets.end()));
}

void check_against_reference(const std::vector<offset_t>& degrees, const host_plan& plan)
{
  const int num_segments = static_cast<int>(degrees.size());

  std::vector<count_t> warp_counts(policy_t::warp_bin_count, 0);
  std::vector<count_t> small_counts(policy_t::small_block_bin_count, 0);
  std::vector<count_t> large_counts(policy_t::large_block_bin_count, 0);
  count_t overflow_count = 0;

  std::vector<std::vector<count_t>> warp_ids(policy_t::warp_bin_count);
  std::vector<std::vector<count_t>> small_ids(policy_t::small_block_bin_count);
  std::vector<std::vector<count_t>> large_ids(policy_t::large_block_bin_count);
  std::vector<count_t> overflow_ids;
  std::vector<count_t> assigned;

  for (int i = 0; i < num_segments; ++i)
  {
    const lrb_shape shape = expected_shape(degrees[static_cast<std::size_t>(i)]);
    if (!shape.assigned())
    {
      continue;
    }
    assigned.push_back(static_cast<count_t>(i));
    if (shape.tier == lrb_tier::warp)
    {
      warp_counts[static_cast<std::size_t>(shape.bin)]++;
      warp_ids[static_cast<std::size_t>(shape.bin)].push_back(static_cast<count_t>(i));
    }
    else if (shape.tier == lrb_tier::small_block)
    {
      small_counts[static_cast<std::size_t>(shape.bin)]++;
      small_ids[static_cast<std::size_t>(shape.bin)].push_back(static_cast<count_t>(i));
    }
    else if (shape.tier == lrb_tier::large_block)
    {
      large_counts[static_cast<std::size_t>(shape.bin)]++;
      large_ids[static_cast<std::size_t>(shape.bin)].push_back(static_cast<count_t>(i));
    }
    else
    {
      overflow_count++;
      overflow_ids.push_back(static_cast<count_t>(i));
    }
  }

  REQUIRE(plan.summary.num_warp == std::accumulate(warp_counts.begin(), warp_counts.end(), count_t{0}));
  REQUIRE(plan.summary.num_small_block == std::accumulate(small_counts.begin(), small_counts.end(), count_t{0}));
  REQUIRE(plan.summary.num_large_block == std::accumulate(large_counts.begin(), large_counts.end(), count_t{0}));
  REQUIRE(plan.summary.num_overflow == overflow_count);
  REQUIRE(plan.summary.num_assigned == assigned.size());

  REQUIRE(plan.warp_offsets.front() == 0);
  REQUIRE(plan.warp_offsets.back() == plan.summary.num_warp);
  REQUIRE(plan.small_offsets.front() == plan.summary.num_warp);
  REQUIRE(plan.small_offsets.back() == plan.summary.num_warp + plan.summary.num_small_block);
  REQUIRE(plan.large_offsets.front() == plan.small_offsets.back());
  REQUIRE(plan.large_offsets.back() == plan.small_offsets.back() + plan.summary.num_large_block);
  REQUIRE(plan.overflow_offsets.front() == plan.large_offsets.back());
  REQUIRE(plan.overflow_offsets.back() == plan.summary.num_assigned);

  for (int i = 0; i < policy_t::warp_bin_count; ++i)
  {
    REQUIRE(plan.warp_offsets[static_cast<std::size_t>(i) + 1] - plan.warp_offsets[static_cast<std::size_t>(i)]
            == warp_counts[static_cast<std::size_t>(i)]);
  }
  for (int i = 0; i < policy_t::small_block_bin_count; ++i)
  {
    REQUIRE(plan.small_offsets[static_cast<std::size_t>(i) + 1] - plan.small_offsets[static_cast<std::size_t>(i)]
            == small_counts[static_cast<std::size_t>(i)]);
  }
  for (int i = 0; i < policy_t::large_block_bin_count; ++i)
  {
    REQUIRE(plan.large_offsets[static_cast<std::size_t>(i) + 1] - plan.large_offsets[static_cast<std::size_t>(i)]
            == large_counts[static_cast<std::size_t>(i)]);
  }

  auto bin_ids = [](const std::vector<count_t>& ids, count_t begin, count_t end) {
    std::vector<count_t> slice(ids.begin() + begin, ids.begin() + end);
    std::sort(slice.begin(), slice.end());
    return slice;
  };

  for (int i = 0; i < policy_t::warp_bin_count; ++i)
  {
    auto got = bin_ids(plan.segment_ids, plan.warp_offsets[i], plan.warp_offsets[i + 1]);
    auto exp = warp_ids[static_cast<std::size_t>(i)];
    std::sort(exp.begin(), exp.end());
    REQUIRE(got == exp);
  }
  for (int i = 0; i < policy_t::small_block_bin_count; ++i)
  {
    auto got = bin_ids(plan.segment_ids, plan.small_offsets[i], plan.small_offsets[i + 1]);
    auto exp = small_ids[static_cast<std::size_t>(i)];
    std::sort(exp.begin(), exp.end());
    REQUIRE(got == exp);
  }
  for (int i = 0; i < policy_t::large_block_bin_count; ++i)
  {
    auto got = bin_ids(plan.segment_ids, plan.large_offsets[i], plan.large_offsets[i + 1]);
    auto exp = large_ids[static_cast<std::size_t>(i)];
    std::sort(exp.begin(), exp.end());
    REQUIRE(got == exp);
  }
  {
    auto got = bin_ids(plan.segment_ids, plan.overflow_offsets[0], plan.overflow_offsets[1]);
    std::sort(overflow_ids.begin(), overflow_ids.end());
    REQUIRE(got == overflow_ids);
  }

  std::vector<count_t> got_ids = plan.segment_ids;
  std::sort(got_ids.begin(), got_ids.end());
  std::sort(assigned.begin(), assigned.end());
  REQUIRE(got_ids == assigned);
  REQUIRE(std::adjacent_find(got_ids.begin(), got_ids.end()) == got_ids.end());

  for (int i = 0; i < policy_t::warp_bin_count; ++i)
  {
    const int threads        = 1 << (i / policy_t::ipt_choices);
    const int segs_per_group = policy_t::warp_threads / threads;
    const count_t groups =
      static_cast<count_t>((warp_counts[static_cast<std::size_t>(i)] + segs_per_group - 1) / segs_per_group);
    REQUIRE(plan.warp_groups[static_cast<std::size_t>(i) + 1] - plan.warp_groups[static_cast<std::size_t>(i)] == groups);
  }
}

CUB_TEST("LRB map: empty and singleton segments are not assigned", "[lrb][segmented][sort][device]", CUB_SMALL)
{
  REQUIRE_FALSE(expected_shape(offset_t{0}).assigned());
  REQUIRE_FALSE(expected_shape(offset_t{1}).assigned());
  REQUIRE(expected_shape(offset_t{2}).assigned());
}

CUB_TEST("LRB map: representative (T, IPT) cells", "[lrb][segmented][sort][device]", CUB_SMALL)
{
  const lrb_shape d2 = expected_shape(offset_t{2});
  REQUIRE(d2.tier == lrb_tier::warp);
  REQUIRE(d2.threads == 1);
  REQUIRE(d2.items_per_thread == 9);

  const lrb_shape d36 = expected_shape(offset_t{36});
  REQUIRE(d36.tier == lrb_tier::warp);
  REQUIRE(d36.threads == 4);
  REQUIRE(d36.items_per_thread == 9);

  const lrb_shape d480 = expected_shape(offset_t{480});
  REQUIRE(d480.tier == lrb_tier::warp);
  REQUIRE(d480.threads == 32);
  REQUIRE(d480.items_per_thread == 15);

  const lrb_shape d481 = expected_shape(offset_t{481});
  REQUIRE(d481.tier == lrb_tier::small_block);
  REQUIRE(d481.threads == 64);
  REQUIRE(d481.items_per_thread == 9);

  const lrb_shape d7680 = expected_shape(offset_t{7680});
  REQUIRE(d7680.tier == lrb_tier::large_block);
  REQUIRE(d7680.threads == 512);
  REQUIRE(d7680.items_per_thread == 15);

  const lrb_shape d15360 = expected_shape(offset_t{15360});
  REQUIRE(d15360.tier == lrb_tier::large_block);
  REQUIRE(d15360.threads == 1024);
  REQUIRE(d15360.items_per_thread == 15);

  const lrb_shape d15361 = expected_shape(offset_t{15361});
  REQUIRE(d15361.tier == lrb_tier::overflow);
}

CUB_TEST("LRB planner: no segments", "[lrb][segmented][sort][device]", CUB_SMALL)
{
  const host_plan plan = run_planner_from_degrees({});
  REQUIRE(plan.summary.num_assigned == 0);
}

CUB_TEST("LRB planner: empty and singleton segments are omitted", "[lrb][segmented][sort][device]", CUB_SMALL)
{
  const std::vector<offset_t> degrees{0, 1, 0, 1};
  const host_plan plan = run_planner_from_degrees(degrees);
  REQUIRE(plan.summary.num_assigned == 0);
  check_against_reference(degrees, plan);
}

CUB_TEST("LRB planner: uniform warp band", "[lrb][segmented][sort][device]", CUB_SMALL)
{
  const std::vector<offset_t> degrees(128, 36);
  check_against_reference(degrees, run_planner_from_degrees(degrees));
}

CUB_TEST("LRB planner: mixed warp, block, overflow, and empties", "[lrb][segmented][sort][device]", CUB_SMALL)
{
  const std::vector<offset_t> degrees{
    0, 1, 2, 9, 16, 36, 60, 72, 288, 481, 2000, 4000, 7680, 10000, 15360, 20000, 3, 1, 0, 512};
  check_against_reference(degrees, run_planner_from_degrees(degrees));
}

CUB_TEST("LRB planner: independent begin/end iterators", "[lrb][segmented][sort][device]", CUB_SMALL)
{
  // Non-CSR: segments occupy disjoint ranges that are not a packed exclusive prefix.
  const std::vector<offset_t> begin{10, 100, 1000, 5000};
  const std::vector<offset_t> end{12, 100, 1036, 5001};
  const std::vector<offset_t> degrees{2, 0, 36, 1};
  const host_plan plan = run_planner(begin, end);
  check_against_reference(degrees, plan);
}
