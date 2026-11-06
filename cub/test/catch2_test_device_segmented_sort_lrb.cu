// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Focused checks for capacity mapping, ID partitioning, and the internal bounded GroupMerge sorting path.

#include <cub/agent/agent_segmented_sort_tiny.cuh>
#include <cub/device/dispatch/dispatch_segmented_sort_lrb.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort_group_merge.cuh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "cub_test_macros.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/device_and_stream.h>

namespace capacity    = cub::detail::segmented_sort_capacity;
constexpr int sm90_cc = cuda::compute_capability{9, 0}.get();

// Irregular capacities exercise bin selection independently of rectangular tuning.
struct irregular_expansion
{
  using capacities = capacity::capacity_list<36, 80, 144>;
  template <int Bin>
  using shape_for_bin = cuda::std::tuple_element_t<
    Bin,
    cuda::std::tuple<capacity::bin_shape<4, 9>, capacity::bin_shape<8, 10>, capacity::bin_shape<16, 9>>>;
};

// One block sorts every segment, one per logical warp, over as many passes as needed.
template <int LogicalWarpThreads, typename Key>
__global__ void tiny_sort_kernel(const int* begins, const int* ends, unsigned num_segments, const Key* in, Key* out)
{
  using sorter_t = capacity::tiny_sort_implementation<Key>;
  __shared__ typename sorter_t::temp_storage storage;
  constexpr unsigned segments_per_pass = capacity::tiny_walk_policy::block_threads / LogicalWarpThreads;
  for (unsigned pass_begin = 0; pass_begin < num_segments; pass_begin += segments_per_pass)
  {
    const unsigned segment_id = pass_begin + threadIdx.x / LogicalWarpThreads;
    const bool is_valid       = segment_id < num_segments;
    sorter_t::template sort_segment<LogicalWarpThreads>(
      storage, is_valid ? segment_id : 0, is_valid, begins, ends, in, out);
    __syncwarp();
  }
}

// Check every length up to each logical warp's capacity, with partial final passes and untouched gaps.
CUB_TEST("Tiny sort sorts each segment within its logical warp capacity",
         "[segmented][sort][device]",
         CUB_SMALL,
         c2h::type_list<int, double>)
{
  using key_type   = c2h::get<0, TestType>;
  const auto check = [](auto logical_warp_threads) {
    constexpr int threads    = decltype(logical_warp_threads)::value;
    constexpr int max_length = threads * capacity::tiny_walk_policy::items_per_thread;
    c2h::host_vector<int> begins, ends;
    c2h::host_vector<key_type> input;
    std::mt19937 random{threads};
    std::uniform_int_distribution<int> random_key{-1000, 1000};
    for (int repeat = 0; repeat < 3; ++repeat)
    {
      for (int length = 0; length <= max_length; ++length)
      {
        input.push_back(key_type{-99});
        begins.push_back(static_cast<int>(input.size()));
        for (int i = 0; i < length; ++i)
        {
          const int value = repeat == 1 ? random_key(random) % 5 : random_key(random);
          input.push_back(static_cast<key_type>(value) / key_type{4});
        }
        ends.push_back(static_cast<int>(input.size()));
      }
    }
    auto expected = input;
    for (size_t segment = 0; segment < begins.size(); ++segment)
    {
      std::sort(expected.begin() + begins[segment], expected.begin() + ends[segment]);
    }
    c2h::device_vector<key_type> d_in = input, d_out(input.size(), key_type{-99});
    c2h::device_vector<int> d_begins = begins, d_ends = ends;
    tiny_sort_kernel<threads><<<1, capacity::tiny_walk_policy::block_threads>>>(
      thrust::raw_pointer_cast(d_begins.data()),
      thrust::raw_pointer_cast(d_ends.data()),
      static_cast<unsigned>(begins.size()),
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_out.data()));
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(c2h::host_vector<key_type>(d_out) == expected);
  };
  check(cuda::std::integral_constant<int, 1>{});
  check(cuda::std::integral_constant<int, 2>{});
  check(cuda::std::integral_constant<int, 4>{});
}

struct tiny_walk_result
{
  c2h::host_vector<unsigned> counts;
  unsigned deferred_count;
  c2h::host_vector<unsigned> deferred_ids;
};

template <typename Capacities, typename Key>
tiny_walk_result
run_tiny_walk(const c2h::host_vector<unsigned>& begins, const c2h::host_vector<unsigned>& ends, const Key* in, Key* out)
{
  using policy_t                        = capacity::tiny_walk_policy;
  const auto num_segments               = static_cast<unsigned>(begins.size());
  constexpr unsigned tile_segments      = policy_t::block_threads * policy_t::segments_per_thread;
  c2h::device_vector<unsigned> d_begins = begins, d_ends = ends;
  c2h::device_vector<unsigned> d_counts(Capacities::size + 1, 0), d_deferred_count(1, 0);
  c2h::device_vector<unsigned> d_deferred_ids(num_segments, ~0u);
  capacity::tiny_walk_kernel<Capacities>
    <<<(num_segments + tile_segments - 1) / tile_segments, policy_t::block_threads>>>(
      thrust::raw_pointer_cast(d_begins.data()),
      thrust::raw_pointer_cast(d_ends.data()),
      num_segments,
      in,
      out,
      thrust::raw_pointer_cast(d_counts.data()),
      thrust::raw_pointer_cast(d_deferred_count.data()),
      thrust::raw_pointer_cast(d_deferred_ids.data()));
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  return {d_counts, c2h::host_vector<unsigned>(d_deferred_count)[0], d_deferred_ids};
}

// The walk's counts size the bins; scattering its queue must fill each bin with exactly the deferred segments
// first_fit assigns to it, advance each cursor to the bin's end, and leave ID slots past the last bin unwritten.
template <typename Capacities>
void check_compact_scatter(const c2h::host_vector<unsigned>& begins,
                           const c2h::host_vector<unsigned>& ends,
                           const tiny_walk_result& walk,
                           unsigned grid_size)
{
  const auto num_segments               = static_cast<unsigned>(begins.size());
  c2h::device_vector<unsigned> d_begins = begins, d_ends = ends, d_counts = walk.counts;
  c2h::device_vector<unsigned> d_deferred_count(1, walk.deferred_count), d_deferred_ids = walk.deferred_ids;
  c2h::device_vector<unsigned> d_offsets(Capacities::size + 2), d_cursors(Capacities::size + 1);
  c2h::device_vector<unsigned> d_ids(num_segments, ~0u);
  capacity::segment_prefix_kernel<Capacities>
    <<<1, 1>>>(thrust::raw_pointer_cast(d_counts.data()),
               thrust::raw_pointer_cast(d_offsets.data()),
               thrust::raw_pointer_cast(d_cursors.data()));
  capacity::compact_scatter_kernel<Capacities><<<grid_size, capacity::partition_policy::block_threads>>>(
    thrust::raw_pointer_cast(d_begins.data()),
    thrust::raw_pointer_cast(d_ends.data()),
    thrust::raw_pointer_cast(d_deferred_count.data()),
    thrust::raw_pointer_cast(d_deferred_ids.data()),
    thrust::raw_pointer_cast(d_cursors.data()),
    thrust::raw_pointer_cast(d_ids.data()));
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const c2h::host_vector<unsigned> offsets = d_offsets;
  const c2h::host_vector<unsigned> cursors = d_cursors;
  const c2h::host_vector<unsigned> ids     = d_ids;
  for (int bin_idx = 0; bin_idx <= Capacities::size; ++bin_idx)
  {
    std::vector<unsigned> bin_ids(ids.begin() + offsets[bin_idx], ids.begin() + offsets[bin_idx + 1]);
    std::vector<unsigned> expected_ids;
    for (unsigned segment = 0; segment < num_segments; ++segment)
    {
      if (ends[segment] > begins[segment]
          && ends[segment] - begins[segment] >= unsigned{capacity::tiny_walk_policy::inline_cutoff}
          && Capacities::first_fit(ends[segment] - begins[segment]) == bin_idx)
      {
        expected_ids.push_back(segment);
      }
    }
    std::sort(bin_ids.begin(), bin_ids.end());
    REQUIRE(bin_ids == expected_ids);
    REQUIRE(cursors[bin_idx] == offsets[bin_idx + 1]);
  }
  REQUIRE(std::all_of(ids.begin() + offsets.back(), ids.end(), [](unsigned id) {
    return id == ~0u;
  }));
}

// Tiny segments and singletons are written and never queued; deferred segments are counted and queued, then
// scattered into their capacity bins, and their keys are not written.
CUB_TEST("Tiny walk sorts short segments and bins deferred ones by capacity",
         "[segmented][sort][device]",
         CUB_SMALL,
         c2h::type_list<int, double>)
{
  using key_type   = c2h::get<0, TestType>;
  using capacities = irregular_expansion::capacities;
  using policy_t   = capacity::tiny_walk_policy;
  std::vector<int> lengths{0, 1, 2, 9, 10, 18, 19, 35, 36, 37, 80, 81, 144, 145};
  std::mt19937 random{12345};
  // Whole tiles of tiny segments skip the count flush; mixed tiles cover several blocks.
  std::uniform_int_distribution<int> tiny_length{0, policy_t::inline_cutoff - 1};
  std::uniform_int_distribution<int> any_length{0, 160};
  for (int i = 0; i < 300; ++i)
  {
    lengths.push_back(tiny_length(random));
  }
  for (int i = 0; i < 600; ++i)
  {
    lengths.push_back(any_length(random));
  }
  // Unsigned offsets: a reversed range must stay empty rather than wrap to a huge length.
  c2h::host_vector<unsigned> begins, ends;
  c2h::host_vector<key_type> input;
  std::uniform_int_distribution<int> random_key{-1000, 1000};
  for (const int length : lengths)
  {
    input.push_back(key_type{-99});
    begins.push_back(static_cast<unsigned>(input.size()));
    for (int i = 0; i < length; ++i)
    {
      input.push_back(static_cast<key_type>(random_key(random)) / key_type{4});
    }
    ends.push_back(static_cast<unsigned>(input.size()));
  }
  lengths.push_back(0);
  begins.push_back(ends.back());
  ends.push_back(ends.back() - 5);
  c2h::host_vector<key_type> expected(input.size(), key_type{-99});
  c2h::host_vector<unsigned> expected_counts(capacities::size + 1, 0);
  for (size_t segment = 0; segment < lengths.size(); ++segment)
  {
    if (ends[segment] <= begins[segment])
    {
      continue;
    }
    if (lengths[segment] >= policy_t::inline_cutoff)
    {
      ++expected_counts[capacities::first_fit(lengths[segment])];
      continue;
    }
    std::copy(input.begin() + begins[segment], input.begin() + ends[segment], expected.begin() + begins[segment]);
    std::sort(expected.begin() + begins[segment], expected.begin() + ends[segment]);
  }
  c2h::device_vector<key_type> d_in = input, d_out(input.size(), key_type{-99});
  const auto num_segments           = static_cast<unsigned>(lengths.size());
  const auto walk                   = run_tiny_walk<capacities>(
    begins, ends, thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(c2h::host_vector<key_type>(d_out) == expected);
  REQUIRE(walk.counts == expected_counts);
  std::vector<unsigned> deferred(walk.deferred_ids.begin(), walk.deferred_ids.begin() + walk.deferred_count);
  std::sort(deferred.begin(), deferred.end());
  std::vector<unsigned> expected_deferred;
  for (unsigned segment = 0; segment < num_segments; ++segment)
  {
    if (lengths[segment] >= policy_t::inline_cutoff)
    {
      expected_deferred.push_back(segment);
    }
  }
  REQUIRE(deferred == expected_deferred);
  REQUIRE(std::all_of(walk.deferred_ids.begin() + walk.deferred_count, walk.deferred_ids.end(), [](unsigned id) {
    return id == ~0u;
  }));
  check_compact_scatter<capacities>(begins, ends, walk, 1);
}

// Deferred segments from many walk blocks arrive in atomic order; small scatter grids stride over the queue
// many times.
CUB_TEST("Compact scatter bins a scattered deferred queue", "[segmented][sort][device]", CUB_SMALL)
{
  using capacities = irregular_expansion::capacities;
  std::mt19937 random{777};
  std::uniform_int_distribution<int> any_length{0, 160};
  c2h::host_vector<unsigned> begins, ends;
  unsigned num_keys = 0;
  for (int i = 0; i < 20000; ++i)
  {
    // Mostly deferred segments, with tiny ones interleaved.
    const int length = i % 4 == 0 ? any_length(random) % capacity::tiny_walk_policy::inline_cutoff
                                  : capacity::tiny_walk_policy::inline_cutoff + any_length(random);
    begins.push_back(num_keys);
    num_keys += length;
    ends.push_back(num_keys);
  }
  c2h::device_vector<int> d_in(num_keys, 0), d_out(num_keys, 0);
  const auto walk = run_tiny_walk<capacities>(
    begins, ends, thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(walk.deferred_count == 15000);
  for (const unsigned grid_size : {1u, 3u, 64u})
  {
    check_compact_scatter<capacities>(begins, ends, walk, grid_size);
  }
}

// Check generic bounds, irregular configuration mapping, sorted tiny segments, untouched overflow, and
// empty-workspace reuse.
CUB_TEST("Tiny walk dispatch bins deferred segments by arbitrary bounds", "[segmented][sort][device]", CUB_SMALL)
{
  for (int bin_idx = 0; bin_idx < 3; ++bin_idx)
  {
    const auto config = capacity::lookup_bin_config<irregular_expansion>(bin_idx);
    REQUIRE(config.threads_per_segment == (4 << bin_idx));
    REQUIRE(config.items_per_thread == (bin_idx == 1 ? 10 : 9));
  }
  static_assert(
    capacity::group_merge_tuning_arch_selector{}(cuda::compute_capability{8, 9}) == cuda::compute_capability{8, 9});
  static_assert(
    capacity::group_merge_tuning_arch_selector{}(cuda::compute_capability{9, 0}) == cuda::compute_capability{9, 0});
  static_assert(
    capacity::group_merge_tuning_arch_selector{}(cuda::compute_capability{12, 0}) == cuda::compute_capability{12, 0});
  static_assert(capacity::tiny_walk_policy::inline_cutoff == 36);
  using capacities = capacity::capacity_list<3, 10, 21>;
  static_assert(capacities::first_fit(3) == 0 && capacities::first_fit(4) == 1);
  static_assert(capacities::first_fit(21) == 2 && capacities::first_fit(22) == 3);
  using walk_capacities = irregular_expansion::capacities;
  // A reversed range first, then lengths around the cutoff and each bound; the last one overflows.
  const std::vector<int> lengths{0, 1, 2, 35, 36, 37, 80, 81, 144, 145};
  c2h::host_vector<int> begins{5}, ends{4}, input(5, -99);
  for (const int length : lengths)
  {
    input.push_back(-99);
    begins.push_back(static_cast<int>(input.size()));
    for (int i = 0; i < length; ++i)
    {
      input.push_back(length - i);
    }
    ends.push_back(static_cast<int>(input.size()));
  }
  auto expected = c2h::host_vector<int>(input.size(), -99);
  for (size_t segment = 0; segment < begins.size(); ++segment)
  {
    if (ends[segment] - begins[segment] < capacity::tiny_walk_policy::inline_cutoff)
    {
      for (int i = begins[segment]; i < ends[segment]; ++i)
      {
        expected[i] = i - begins[segment] + 1;
      }
    }
  }
  c2h::device_vector<int> d_begins = begins, d_ends = ends, d_in = input, d_out(input.size(), -99);
  capacity::partition_result result;
  size_t bytes   = 0;
  const auto run = [&](void* storage, unsigned count) {
    return capacity::dispatch_tiny_sort_and_partition<walk_capacities>(
      storage,
      bytes,
      thrust::raw_pointer_cast(d_begins.data()),
      thrust::raw_pointer_cast(d_ends.data()),
      static_cast<cuda::std::int64_t>(input.size()),
      count,
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_out.data()),
      result);
  };
  REQUIRE(run(nullptr, static_cast<unsigned>(begins.size())) == cudaSuccess);
  c2h::device_vector<unsigned char> storage(bytes);
  REQUIRE(run(thrust::raw_pointer_cast(storage.data()), static_cast<unsigned>(begins.size())) == cudaSuccess);
  c2h::host_vector<unsigned> offsets(walk_capacities::size + 2);
  REQUIRE(cudaMemcpy(offsets.data(), result.bin_offsets, offsets.size() * sizeof(unsigned), cudaMemcpyDeviceToHost)
          == cudaSuccess);
  REQUIRE(offsets == c2h::host_vector<unsigned>{0, 1, 3, 5, 6});
  REQUIRE(c2h::host_vector<int>(d_out) == expected);
  c2h::host_vector<unsigned> ids(offsets.back());
  REQUIRE(
    cudaMemcpy(ids.data(), result.segment_ids, ids.size() * sizeof(unsigned), cudaMemcpyDeviceToHost) == cudaSuccess);
  for (int bin_idx = 0; bin_idx <= walk_capacities::size; ++bin_idx)
  {
    for (unsigned i = offsets[bin_idx]; i < offsets[bin_idx + 1]; ++i)
    {
      REQUIRE(walk_capacities::first_fit(ends[ids[i]] - begins[ids[i]]) == bin_idx);
    }
  }
  std::sort(ids.begin(), ids.end());
  REQUIRE(ids == c2h::host_vector<unsigned>{5, 6, 7, 8, 9, 10});
  // Reuse the same workspace with no segments; every range must become empty.
  REQUIRE(run(thrust::raw_pointer_cast(storage.data()), 0) == cudaSuccess);
  REQUIRE(cudaMemcpy(offsets.data(), result.bin_offsets, offsets.size() * sizeof(unsigned), cudaMemcpyDeviceToHost)
          == cudaSuccess);
  REQUIRE(offsets == c2h::host_vector<unsigned>(walk_capacities::size + 2, 0));
}

// Non-overlapping segments within num_items keys defer at most num_items / inline_cutoff segments, so both ID
// buffers are sized by that bound, and the bound is reached when every segment has exactly the cutoff length.
CUB_TEST("Tiny walk dispatch bounds ID buffers by the number of items", "[segmented][sort][device]", CUB_SMALL)
{
  using capacities      = irregular_expansion::capacities;
  constexpr int cutoff  = capacity::tiny_walk_policy::inline_cutoff;
  const auto query_size = [](cuda::std::int64_t num_items, unsigned num_segments) {
    size_t bytes = 0;
    capacity::partition_result result;
    REQUIRE(
      capacity::dispatch_tiny_sort_and_partition<capacities>(
        nullptr,
        bytes,
        static_cast<const int*>(nullptr),
        static_cast<const int*>(nullptr),
        num_items,
        num_segments,
        static_cast<const int*>(nullptr),
        static_cast<int*>(nullptr),
        result)
      == cudaSuccess);
    return bytes;
  };
  REQUIRE(query_size(10 * cutoff, 1000000) == query_size(10 * cutoff, 10));
  REQUIRE(query_size(10 * cutoff + cutoff - 1, 1000000) == query_size(10 * cutoff, 10));
  REQUIRE(query_size(1000000 * cutoff, 10) == query_size(10 * cutoff, 10));
  REQUIRE(query_size(1000000 * cutoff, 1000000) > query_size(10 * cutoff, 10));
  REQUIRE(query_size(cutoff - 1, 1000) == query_size(0, 0));
  REQUIRE(query_size(-1, 1000) == query_size(0, 0));

  const auto run =
    [](const c2h::host_vector<int>& begins,
       const c2h::host_vector<int>& ends,
       c2h::device_vector<int>& d_in,
       c2h::device_vector<int>& d_out) {
      c2h::device_vector<int> d_begins = begins, d_ends = ends;
      const auto num_items    = static_cast<cuda::std::int64_t>(d_in.size());
      const auto num_segments = static_cast<unsigned>(begins.size());
      capacity::partition_result result;
      size_t bytes      = 0;
      const auto launch = [&](void* storage) {
        return capacity::dispatch_tiny_sort_and_partition<capacities>(
          storage,
          bytes,
          thrust::raw_pointer_cast(d_begins.data()),
          thrust::raw_pointer_cast(d_ends.data()),
          num_items,
          num_segments,
          thrust::raw_pointer_cast(d_in.data()),
          thrust::raw_pointer_cast(d_out.data()),
          result);
      };
      REQUIRE(launch(nullptr) == cudaSuccess);
      c2h::device_vector<unsigned char> storage(bytes);
      REQUIRE(launch(thrust::raw_pointer_cast(storage.data())) == cudaSuccess);
      REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
      c2h::host_vector<unsigned> offsets(capacities::size + 2);
      REQUIRE(cudaMemcpy(offsets.data(), result.bin_offsets, offsets.size() * sizeof(unsigned), cudaMemcpyDeviceToHost)
              == cudaSuccess);
      c2h::host_vector<unsigned> ids(offsets.back());
      REQUIRE(cudaMemcpy(ids.data(), result.segment_ids, ids.size() * sizeof(unsigned), cudaMemcpyDeviceToHost)
              == cudaSuccess);
      return std::pair{offsets, ids};
    };

  // Every segment is deferred, so both buffers fill to exactly num_items / cutoff entries; the trailing keys
  // belong to no segment. num_full - 1 IDs fill whole 256-byte blocks, so a bound one short shrinks the storage.
  constexpr int num_full = 3009;
  REQUIRE(query_size(num_full * cutoff + cutoff - 1, num_full) - query_size(0, 0) >= 2 * sizeof(unsigned) * num_full);
  c2h::host_vector<int> begins(num_full), ends(num_full);
  for (int segment = 0; segment < num_full; ++segment)
  {
    begins[segment] = segment * cutoff;
    ends[segment]   = begins[segment] + cutoff;
  }
  c2h::device_vector<int> d_in(num_full * cutoff + cutoff - 1, 1), d_out(d_in.size(), 0);
  const auto [offsets, ids] = run(begins, ends, d_in, d_out);
  REQUIRE(offsets == c2h::host_vector<unsigned>{0, num_full, num_full, num_full, num_full});
  auto sorted_ids = ids;
  std::sort(sorted_ids.begin(), sorted_ids.end());
  c2h::host_vector<unsigned> expected_ids(num_full);
  std::iota(expected_ids.begin(), expected_ids.end(), 0u);
  REQUIRE(sorted_ids == expected_ids);

  // Fewer keys than the cutoff: no segment can be deferred and the ID buffers are empty, but tiny segments
  // are still sorted.
  const c2h::host_vector<int> tiny_begins{0, 1, 3, 20}, tiny_ends{1, 3, 20, 35};
  c2h::host_vector<int> tiny_input(cutoff - 1);
  for (int i = 0; i < cutoff - 1; ++i)
  {
    tiny_input[i] = cutoff - i;
  }
  c2h::host_vector<int> tiny_expected = tiny_input;
  for (size_t segment = 0; segment < tiny_begins.size(); ++segment)
  {
    std::sort(tiny_expected.begin() + tiny_begins[segment], tiny_expected.begin() + tiny_ends[segment]);
  }
  c2h::device_vector<int> d_tiny_in   = tiny_input, d_tiny_out(tiny_input.size(), 0);
  const auto [tiny_offsets, tiny_ids] = run(tiny_begins, tiny_ends, d_tiny_in, d_tiny_out);
  REQUIRE(tiny_offsets == c2h::host_vector<unsigned>(capacities::size + 2, 0));
  REQUIRE(c2h::host_vector<int>(d_tiny_out) == tiny_expected);
}

// Exercise capacity boundaries and all launch groups, including repeated work and untouched overflow.
CUB_TEST("Capacity sort reconstructs work prefixes in each CTA",
         "[segmented][sort][device]",
         CUB_SMALL,
         c2h::type_list<int, double>)
{
  using key_type             = c2h::get<0, TestType>;
  using kernels_t            = capacity::group_merge_sort_kernels_t<key_type>;
  using capacities           = typename kernels_t::template capacities_t<cub::detail::policy_getter, sm90_cc>;
  using warp_kernel_t        = typename kernels_t::template kernel_t<0>;
  using warp_policy_getter_t = cub::detail::policy_getter<typename warp_kernel_t::policy_selector, sm90_cc>;
  using warp_expansion       = typename warp_kernel_t::template expansion_t<warp_policy_getter_t>;
  const int bin_idx          = warp_expansion::capacities::first_fit(37);
  const auto config          = capacity::lookup_bin_config<warp_expansion>(bin_idx);
  REQUIRE(config.threads_per_segment == 4);
  REQUIRE(config.items_per_thread == 10);
  // Walk boundaries: empty, singleton, and the first and last length of each logical warp size. The
  // capacity bounds below add the cutoff (35, 36, 37).
  constexpr int walk_ipt = capacity::tiny_walk_policy::items_per_thread;
  std::vector<int> lengths{0, 1, 2, walk_ipt, walk_ipt + 1, 2 * walk_ipt, 2 * walk_ipt + 1};
  static_assert(capacities::bounds[0] == capacity::tiny_walk_policy::inline_cutoff);
  for (const int bound : capacities::bounds)
  {
    for (const int delta : {-1, 0, 1})
    {
      lengths.push_back(bound + delta);
    }
  }
  // A partially filled final group, many work iterations, and absent bins.
  lengths.insert(lengths.end(), 4097, warp_expansion::capacities::bounds.back());
  using small_kernel_t        = typename kernels_t::template kernel_t<1>;
  using small_policy_getter_t = cub::detail::policy_getter<typename small_kernel_t::policy_selector, sm90_cc>;
  using small_capacities      = typename small_kernel_t::template capacities_t<small_policy_getter_t>;
  lengths.insert(lengths.end(), 1000, small_capacities::bounds.back());
  lengths.insert(lengths.end(), 7, capacities::bounds.back() - 1);
  c2h::host_vector<long long> begins, ends;
  c2h::host_vector<key_type> input, expected;
  std::mt19937 random{12345};
  std::uniform_int_distribution<int> random_key{-100000, 100000};
  for (size_t segment = 0; segment < lengths.size(); ++segment)
  {
    const int length = lengths[segment];
    // Gaps also verify that only each requested range is written.
    input.push_back(key_type{-99});
    begins.push_back(input.size());
    for (int i = 0; i < length; ++i)
    {
      // Alternate random, duplicate-heavy, and descending inputs, including negative doubles.
      const int value = segment % 3 == 0 ? random_key(random)
                      : segment % 3 == 1 ? random_key(random) % 7
                                         : (length - i) % 257 - 128;
      input.push_back(static_cast<key_type>(value) / key_type{4});
    }
    ends.push_back(input.size());
  }
  expected.assign(input.size(), key_type{-99});
  for (size_t segment = 0; segment < lengths.size(); ++segment)
  {
    if (lengths[segment] > capacities::bounds.back())
    {
      continue; // Overflow is classified but its output must remain untouched.
    }
    std::copy(input.begin() + begins[segment], input.begin() + ends[segment], expected.begin() + begins[segment]);
    std::stable_sort(expected.begin() + begins[segment], expected.begin() + ends[segment]);
  }
  c2h::device_vector<key_type> d_in      = input, d_out(input.size(), key_type{-99});
  c2h::device_vector<long long> d_begins = begins, d_ends = ends;
  capacity::partition_result result;
  size_t bytes              = 0;
  const auto stream_owner   = c2h::make_current_device_stream();
  const cudaStream_t stream = stream_owner.get();
  const auto run            = [&](void* storage) {
    return capacity::dispatch(
      storage,
      bytes,
      thrust::raw_pointer_cast(d_begins.data()),
      thrust::raw_pointer_cast(d_ends.data()),
      static_cast<cuda::std::int64_t>(input.size()),
      static_cast<unsigned>(lengths.size()),
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_out.data()),
      result,
      stream);
  };
  const auto query_status = run(nullptr);
  cuda::compute_capability cc{};
  REQUIRE(cub::detail::ptx_compute_cap(cc) == cudaSuccess);
  if (cc < cuda::compute_capability{9, 0})
  {
    REQUIRE(query_status == cudaErrorNotSupported);
    return;
  }
  REQUIRE(query_status == cudaSuccess);
  c2h::device_vector<unsigned char> storage(bytes);
  REQUIRE(run(thrust::raw_pointer_cast(storage.data())) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  REQUIRE(c2h::host_vector<key_type>(d_out) == expected);
  c2h::host_vector<unsigned> offsets(capacities::size + 2);
  REQUIRE(cudaMemcpy(offsets.data(), result.bin_offsets, offsets.size() * sizeof(unsigned), cudaMemcpyDeviceToHost)
          == cudaSuccess);
  REQUIRE(offsets.back() - offsets[capacities::size] == 1);
  REQUIRE(c2h::host_vector<key_type>(d_in) == input);
  // Force scratch reuse and cross-bin work iterations independently in every sorting kernel.
  const auto retry_kernel = [&](auto index) {
    constexpr int i           = decltype(index)::value;
    using kernel_t            = typename kernels_t::template kernel_t<i>;
    using policy_getter_t     = cub::detail::policy_getter<typename kernel_t::policy_selector, sm90_cc>;
    using kernel_capacities_t = typename kernel_t::template capacities_t<policy_getter_t>;
    constexpr int first_bin   = kernels_t::template kernel_bin_begin_idx<sm90_cc>(i);
    // Lengths below the walk's cutoff are never binned, so the warp kernel does not rewrite them.
    constexpr int lower_bound = first_bin ? capacities::bounds[static_cast<size_t>(first_bin - 1)]
                                          : capacity::tiny_walk_policy::inline_cutoff - 1;
    auto retry                = expected;
    for (size_t segment = 0; segment < lengths.size(); ++segment)
    {
      if (lengths[segment] > lower_bound && lengths[segment] <= kernel_capacities_t::bounds.back())
      {
        std::fill(retry.begin() + begins[segment], retry.begin() + ends[segment], key_type{-99});
      }
    }
    d_out = retry;
    REQUIRE(
      capacity::launch_sort_kernel<kernel_t, sm90_cc>(
        {result.bin_offsets + first_bin, result.segment_ids},
        thrust::raw_pointer_cast(d_begins.data()),
        thrust::raw_pointer_cast(d_ends.data()),
        thrust::raw_pointer_cast(d_in.data()),
        thrust::raw_pointer_cast(d_out.data()),
        1,
        stream,
        cub::detail::TripleChevronFactory{})
      == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    REQUIRE(c2h::host_vector<key_type>(d_out) == expected);
  };
  retry_kernel(cuda::std::integral_constant<int, 0>{});
  retry_kernel(cuda::std::integral_constant<int, 1>{});
  retry_kernel(cuda::std::integral_constant<int, 2>{});
  // Prefixes and scatter cursors must reset on a subsequent invocation.
  thrust::fill(d_out.begin(), d_out.end(), key_type{-99});
  REQUIRE(run(thrust::raw_pointer_cast(storage.data())) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  REQUIRE(c2h::host_vector<key_type>(d_out) == expected);
}

// Check supported floating-point comparisons without imposing radix-sort signed-zero ordering.
CUB_TEST("Capacity sort orders infinities and treats signed zeros as equivalent",
         "[segmented][sort][device]",
         CUB_SMALL,
         c2h::type_list<float, double>)
{
  using key_type          = c2h::get<0, TestType>;
  const key_type infinity = std::numeric_limits<key_type>::infinity();
  const c2h::host_vector<key_type> input{infinity, -key_type{0}, 3, -infinity, key_type{0}, -7, -1, 1};
  const c2h::host_vector<key_type> expected{-infinity, -7, -1, 0, 0, 1, 3, infinity};
  c2h::device_vector<key_type> d_in = input, d_out(input.size());
  c2h::device_vector<int> d_offsets = c2h::host_vector<int>{0, static_cast<int>(input.size())};
  const auto stream_owner           = c2h::make_current_device_stream();
  const cudaStream_t stream         = stream_owner.get();
  capacity::partition_result result;
  size_t bytes   = 0;
  const auto run = [&](void* storage) {
    const auto* offsets = thrust::raw_pointer_cast(d_offsets.data());
    return capacity::dispatch(
      storage,
      bytes,
      offsets,
      offsets + 1,
      static_cast<cuda::std::int64_t>(input.size()),
      1,
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_out.data()),
      result,
      stream);
  };
  const auto query_status = run(nullptr);
  cuda::compute_capability cc{};
  REQUIRE(cub::detail::ptx_compute_cap(cc) == cudaSuccess);
  if (cc < cuda::compute_capability{9, 0})
  {
    REQUIRE(query_status == cudaErrorNotSupported);
    return;
  }
  REQUIRE(query_status == cudaSuccess);
  c2h::device_vector<unsigned char> storage(bytes);
  REQUIRE(run(thrust::raw_pointer_cast(storage.data())) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  const c2h::host_vector<key_type> output = d_out;
  REQUIRE(output == expected);
  // Both zero signs are preserved and adjacent; their relative order is unspecified.
  REQUIRE(std::signbit(output[3]) != std::signbit(output[4]));
}
