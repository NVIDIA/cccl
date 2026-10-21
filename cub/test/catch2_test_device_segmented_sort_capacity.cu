// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/dispatch/dispatch_segmented_sort_capacity.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort_group_merge.cuh>

#include <cuda/std/functional>

#include <algorithm>
#include <vector>

#include "cub_test_macros.h"
#include <c2h/catch2_test_helper.h>

namespace capacity = cub::detail::segmented_sort_capacity;

// A small grid exercises persistent work and scratch reuse on either test GPU.
template <typename Key>
struct test_selector : capacity::group_merge_policy_selector<Key>
{
  _CCCL_HOST_DEVICE_API constexpr capacity::group_merge_launch_policy operator()(cuda::compute_capability cc) const
  {
    auto policy               = capacity::group_merge_policy_selector<Key>::operator()(cc);
    policy.warp.blocks_per_sm = policy.small.blocks_per_sm = policy.large.blocks_per_sm = 1;
    return policy;
  }
};

CUB_TEST("Capacity partition accepts arbitrary bounds without sort tuning", "[segmented][sort][device]", CUB_SMALL)
{
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
  using key_type   = c2h::get<0, TestType>;
  using selector   = test_selector<key_type>;
  using getter     = cub::detail::policy_getter<selector, 75>;
  using plan       = typename capacity::group_merge_entries<getter>::type;
  using capacities = typename plan::capacities;
  std::vector<int> lengths{0, 1};
  for (int bound : capacities::bounds)
  {
    for (int delta : {-1, 0, 1})
    {
      lengths.push_back(bound + delta);
    }
  }
  // A partially filled final group, many work iterations, and absent bins.
  lengths.insert(lengths.end(), 4097, 480);
  c2h::host_vector<long long> begins, ends;
  c2h::host_vector<key_type> input, expected;
  for (int length : lengths)
  {
    // Gaps also verify that only each requested range is written.
    input.push_back(key_type{-99});
    begins.push_back(input.size());
    for (int i = 0; i < length; ++i)
    {
      input.push_back(static_cast<key_type>((length - i) % 257 - 128));
    }
    ends.push_back(input.size());
  }
  expected.assign(input.size(), key_type{-99});
  for (size_t segment = 0; segment < lengths.size(); ++segment)
  {
    if (lengths[segment] > capacities::bounds.back())
    {
      continue; // overflow is returned to the caller
    }
    std::copy(input.begin() + begins[segment], input.begin() + ends[segment], expected.begin() + begins[segment]);
    std::stable_sort(expected.begin() + begins[segment], expected.begin() + ends[segment]);
  }
  c2h::device_vector<key_type> d_in      = input, d_out(input.size(), key_type{-99});
  c2h::device_vector<long long> d_begins = begins, d_ends = ends;
  capacity::partition_result result;
  size_t bytes = 0;
  cudaStream_t stream{};
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
  const auto run = [&](void* storage) {
    return capacity::dispatch<capacity::group_merge_entries>(
      storage,
      bytes,
      thrust::raw_pointer_cast(d_begins.data()),
      thrust::raw_pointer_cast(d_ends.data()),
      static_cast<unsigned>(lengths.size()),
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_out.data()),
      cuda::std::less<key_type>{},
      result,
      stream,
      selector{});
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
  // Prefixes and scatter cursors must reset on a subsequent invocation.
  REQUIRE(run(thrust::raw_pointer_cast(storage.data())) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  REQUIRE(c2h::host_vector<key_type>(d_out) == expected);
  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}
