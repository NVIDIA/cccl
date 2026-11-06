// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Focused checks for capacity mapping, ID partitioning, and the internal bounded GroupMerge sorting path.

#include <cub/agent/agent_segmented_sort_tiny.cuh>
#include <cub/device/dispatch/dispatch_segmented_sort_lrb.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort_group_merge.cuh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
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

// Tiny segments and singletons are written and never queued; deferred segments are counted, then scattered
// into their capacity bins, and their keys are not written.
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
  c2h::device_vector<key_type> d_in     = input, d_out(input.size(), key_type{-99});
  c2h::device_vector<unsigned> d_begins = begins, d_ends = ends;
  c2h::device_vector<unsigned> d_counts(capacities::size + 1, 0);
  const auto num_segments          = static_cast<unsigned>(lengths.size());
  constexpr unsigned tile_segments = policy_t::block_threads * policy_t::segments_per_thread;
  capacity::tiny_walk_kernel<capacities>
    <<<(num_segments + tile_segments - 1) / tile_segments, policy_t::block_threads>>>(
      thrust::raw_pointer_cast(d_begins.data()),
      thrust::raw_pointer_cast(d_ends.data()),
      num_segments,
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_out.data()),
      thrust::raw_pointer_cast(d_counts.data()));
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(c2h::host_vector<key_type>(d_out) == expected);
  REQUIRE(c2h::host_vector<unsigned>(d_counts) == expected_counts);
  // The walk's counts size the bins; a scatter with the same cutoff must fill them exactly.
  c2h::device_vector<unsigned> d_offsets(capacities::size + 2), d_cursors(capacities::size + 1);
  c2h::device_vector<unsigned> d_ids(num_segments, ~0u);
  capacity::segment_prefix_kernel<capacities>
    <<<1, 1>>>(thrust::raw_pointer_cast(d_counts.data()),
               thrust::raw_pointer_cast(d_offsets.data()),
               thrust::raw_pointer_cast(d_cursors.data()));
  constexpr unsigned partition_threads = capacity::partition_policy::block_threads;
  capacity::partition_kernel<capacities, true, const unsigned*, const unsigned*, policy_t::inline_cutoff>
    <<<(num_segments + partition_threads - 1) / partition_threads, partition_threads>>>(
      thrust::raw_pointer_cast(d_begins.data()),
      thrust::raw_pointer_cast(d_ends.data()),
      num_segments,
      thrust::raw_pointer_cast(d_cursors.data()),
      thrust::raw_pointer_cast(d_ids.data()));
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const c2h::host_vector<unsigned> offsets = d_offsets;
  const c2h::host_vector<unsigned> ids     = d_ids;
  for (int bin_idx = 0; bin_idx <= capacities::size; ++bin_idx)
  {
    std::vector<unsigned> bin_ids(ids.begin() + offsets[bin_idx], ids.begin() + offsets[bin_idx + 1]);
    std::vector<unsigned> expected_ids;
    for (unsigned segment = 0; segment < num_segments; ++segment)
    {
      if (lengths[segment] >= policy_t::inline_cutoff && capacities::first_fit(lengths[segment]) == bin_idx)
      {
        expected_ids.push_back(segment);
      }
    }
    std::sort(bin_ids.begin(), bin_ids.end());
    REQUIRE(bin_ids == expected_ids);
  }
  REQUIRE(std::all_of(ids.begin() + offsets.back(), ids.end(), [](unsigned id) {
    return id == ~0u;
  }));
}

// Check generic bounds, irregular configuration mapping, overflow IDs, and empty-workspace reuse.
CUB_TEST("Capacity partition accepts arbitrary bounds without sort tuning", "[segmented][sort][device]", CUB_SMALL)
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
  const c2h::host_vector<int> begins{5, 7, 9, 20, 30, 40, 50, 60, 90};
  const c2h::host_vector<int> ends{4, 7, 10, 23, 34, 50, 61, 81, 112};
  c2h::device_vector<int> d_begins = begins, d_ends = ends;
  capacity::partition_result result;
  size_t bytes   = 0;
  const auto run = [&](void* storage, unsigned count) {
    return capacity::dispatch_partition<capacities>(
      storage, bytes, thrust::raw_pointer_cast(d_begins.data()), thrust::raw_pointer_cast(d_ends.data()), count, result);
  };
  REQUIRE(run(nullptr, static_cast<unsigned>(begins.size())) == cudaSuccess);
  c2h::device_vector<unsigned char> storage(bytes);
  REQUIRE(run(thrust::raw_pointer_cast(storage.data()), static_cast<unsigned>(begins.size())) == cudaSuccess);
  c2h::host_vector<unsigned> offsets(capacities::size + 2);
  REQUIRE(cudaMemcpy(offsets.data(), result.bin_offsets, offsets.size() * sizeof(unsigned), cudaMemcpyDeviceToHost)
          == cudaSuccess);
  REQUIRE(offsets == c2h::host_vector<unsigned>{0, 2, 4, 6, 7});
  c2h::host_vector<unsigned> ids(offsets.back());
  REQUIRE(
    cudaMemcpy(ids.data(), result.segment_ids, ids.size() * sizeof(unsigned), cudaMemcpyDeviceToHost) == cudaSuccess);
  for (int bin_idx = 0; bin_idx <= capacities::size; ++bin_idx)
  {
    for (unsigned i = offsets[bin_idx]; i < offsets[bin_idx + 1]; ++i)
    {
      REQUIRE(capacities::first_fit(ends[ids[i]] - begins[ids[i]]) == bin_idx);
    }
  }
  std::sort(ids.begin(), ids.end());
  REQUIRE(ids == c2h::host_vector<unsigned>{2, 3, 4, 5, 6, 7, 8});
  // Reuse the same workspace with no segments; every range must become empty.
  REQUIRE(run(thrust::raw_pointer_cast(storage.data()), 0) == cudaSuccess);
  REQUIRE(cudaMemcpy(offsets.data(), result.bin_offsets, offsets.size() * sizeof(unsigned), cudaMemcpyDeviceToHost)
          == cudaSuccess);
  REQUIRE(offsets == c2h::host_vector<unsigned>(capacities::size + 2, 0));
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
  std::vector<int> lengths{0, 1};
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
    constexpr int lower_bound = first_bin ? capacities::bounds[static_cast<size_t>(first_bin - 1)] : 0;
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
