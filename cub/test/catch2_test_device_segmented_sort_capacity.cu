// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/dispatch/dispatch_segmented_sort_capacity.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort_group_merge.cuh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "cub_test_macros.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/device_and_stream.h>

namespace capacity = cub::detail::segmented_sort_capacity;

// Irregular capacities exercise bin selection independently of rectangular tuning.
struct irregular_expansion
{
  using capacities = capacity::capacity_list<36, 80, 144>;
  template <int Bin>
  using config_for_bin = cuda::std::tuple_element_t<
    Bin,
    cuda::std::tuple<capacity::bin_shape<4, 9>, capacity::bin_shape<8, 10>, capacity::bin_shape<16, 9>>>;
};

CUB_TEST("Capacity partition accepts arbitrary bounds without sort tuning", "[segmented][sort][device]", CUB_SMALL)
{
  for (int bin = 0; bin < 3; ++bin)
  {
    const auto config = capacity::config_for_bin<irregular_expansion>(bin);
    REQUIRE(config.threads == (4 << bin));
    REQUIRE(config.items_per_thread == (bin == 1 ? 10 : 9));
  }
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
  REQUIRE(run(nullptr, begins.size()) == cudaSuccess);
  c2h::device_vector<unsigned char> storage(bytes);
  REQUIRE(run(thrust::raw_pointer_cast(storage.data()), begins.size()) == cudaSuccess);
  c2h::host_vector<unsigned> offsets(capacities::size + 2);
  REQUIRE(cudaMemcpy(offsets.data(), result.bin_offsets, offsets.size() * sizeof(unsigned), cudaMemcpyDeviceToHost)
          == cudaSuccess);
  REQUIRE(offsets == c2h::host_vector<unsigned>{0, 2, 4, 6, 7});
  c2h::host_vector<unsigned> ids(offsets.back());
  REQUIRE(
    cudaMemcpy(ids.data(), result.segment_ids, ids.size() * sizeof(unsigned), cudaMemcpyDeviceToHost) == cudaSuccess);
  for (int bin = 0; bin <= capacities::size; ++bin)
  {
    for (unsigned i = offsets[bin]; i < offsets[bin + 1]; ++i)
    {
      REQUIRE(capacities::first_fit(ends[ids[i]] - begins[ids[i]]) == bin);
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

CUB_TEST("Capacity sort reconstructs work prefixes in each CTA",
         "[segmented][sort][device]",
         CUB_SMALL,
         c2h::type_list<int, double>)
{
  using key_type       = c2h::get<0, TestType>;
  using selector       = capacity::group_merge_policy_selector<key_type>;
  using getter         = cub::detail::policy_getter<selector, 75>;
  using plan           = typename capacity::group_merge_entries<getter>::type;
  using capacities     = typename plan::capacities;
  using warp_expansion = typename plan::template entry<0>::expansion;
  const int bin        = warp_expansion::capacities::first_fit(37);
  const auto config    = capacity::config_for_bin<warp_expansion>(bin);
  REQUIRE(config.threads == 4);
  REQUIRE(config.items_per_thread == 10);
  std::vector<int> lengths{0, 1};
  for (int bound : capacities::bounds)
  {
    for (int delta : {-1, 0, 1})
    {
      lengths.push_back(bound + delta);
    }
  }
  // A partially filled final group, many work iterations, and absent bins.
  lengths.insert(lengths.end(), 4097, warp_expansion::capacities::bounds.back());
  using small_capacities = typename plan::template entry<1>::capacities;
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
  REQUIRE(run(nullptr) == cudaSuccess);
  c2h::device_vector<unsigned char> storage(bytes);
  REQUIRE(run(thrust::raw_pointer_cast(storage.data())) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  REQUIRE(c2h::host_vector<key_type>(d_out) == expected);
  c2h::host_vector<unsigned> offsets(capacities::size + 2);
  REQUIRE(cudaMemcpy(offsets.data(), result.bin_offsets, offsets.size() * sizeof(unsigned), cudaMemcpyDeviceToHost)
          == cudaSuccess);
  REQUIRE(offsets.back() - offsets[capacities::size] == 1);
  REQUIRE(c2h::host_vector<key_type>(d_in) == input);
  // Force scratch reuse and cross-bin work iterations independently in every entry.
  const auto retry_entry = [&](auto index) {
    constexpr int i           = decltype(index)::value;
    using entry               = typename plan::template entry<i>;
    constexpr int first_bin   = plan::bin_begin(i);
    constexpr int lower_bound = first_bin ? capacities::bounds[first_bin - 1] : 0;
    auto retry                = expected;
    for (size_t segment = 0; segment < lengths.size(); ++segment)
    {
      if (lengths[segment] > lower_bound && lengths[segment] <= entry::capacities::bounds.back())
      {
        std::fill(retry.begin() + begins[segment], retry.begin() + ends[segment], key_type{-99});
      }
    }
    d_out = retry;
    REQUIRE(
      capacity::launch_entry<entry>(
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
  retry_entry(cuda::std::integral_constant<int, 0>{});
  retry_entry(cuda::std::integral_constant<int, 1>{});
  retry_entry(cuda::std::integral_constant<int, 2>{});
  // Prefixes and scatter cursors must reset on a subsequent invocation.
  thrust::fill(d_out.begin(), d_out.end(), key_type{-99});
  REQUIRE(run(thrust::raw_pointer_cast(storage.data())) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  REQUIRE(c2h::host_vector<key_type>(d_out) == expected);
}

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
  REQUIRE(run(nullptr) == cudaSuccess);
  c2h::device_vector<unsigned char> storage(bytes);
  REQUIRE(run(thrust::raw_pointer_cast(storage.data())) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  const c2h::host_vector<key_type> output = d_out;
  REQUIRE(output == expected);
  // Both zero signs are preserved and adjacent; their relative order is unspecified.
  REQUIRE(std::signbit(output[3]) != std::signbit(output[4]));
}
