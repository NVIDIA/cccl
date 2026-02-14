// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_reduce.cuh>

#include <thrust/device_vector.h>
#include <thrust/equal.h>

#include <cuda/std/utility>

#include <climits>
#include <cstddef>

#include "thrust/detail/raw_pointer_cast.h"
#include <c2h/catch2_test_helper.h>

// example-begin segmented-reduce-custommin
struct CustomMin
{
  template <typename T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const
  {
    return (b < a) ? b : a;
  }
};

// example-end segmented-reduce-custommin

struct is_equal
{
  __device__ bool operator()(cub::KeyValuePair<int, int> lhs, cub::KeyValuePair<int, int> rhs)
  {
    return !(lhs != rhs);
  }

  __device__ bool operator()(cuda::std::pair<int, int> lhs, cuda::std::pair<int, int> rhs)
  {
    return !(lhs != rhs);
  }
};

C2H_TEST("cub::DeviceSegmentedReduce::Reduce works with int data elements", "[segmented_reduce][device]")
{
  // example-begin segmented-reduce-reduce
  int num_segments                  = 3;
  c2h::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                 = thrust::raw_pointer_cast(d_offsets.data());
  c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<int> d_out(3);
  CustomMin min_op;
  int initial_value{INT_MAX};

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Reduce(
    d_temp_storage,
    temp_storage_bytes,
    d_in.begin(),
    d_out.begin(),
    num_segments,
    d_offsets_it,
    d_offsets_it + 1,
    min_op,
    initial_value);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::Reduce(
    d_temp_storage,
    temp_storage_bytes,
    d_in.begin(),
    d_out.begin(),
    num_segments,
    d_offsets_it,
    d_offsets_it + 1,
    min_op,
    initial_value);

  c2h::device_vector<int> expected{6, INT_MAX, 0};
  // example-end segmented-reduce-reduce

  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedReduce::Sum works with int data elements", "[segmented_reduce][device]")
{
  // example-begin segmented-reduce-sum
  int num_segments                  = 3;
  c2h::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                 = thrust::raw_pointer_cast(d_offsets.data());
  c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<int> d_out(3);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::Sum(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  c2h::device_vector<int> expected{21, 0, 17};
  // example-end segmented-reduce-sum

  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedReduce::Min works with int data elements", "[segmented_reduce][device]")
{
  // example-begin segmented-reduce-min
  int num_segments                  = 3;
  c2h::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                 = thrust::raw_pointer_cast(d_offsets.data());
  c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<int> d_out(3);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Min(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::Min(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  c2h::device_vector<int> expected{6, INT_MAX, 0};
  // example-end segmented-reduce-min

  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedReduce::ArgMin works with int data elements", "[segmented_reduce][device]")
{
  // example-begin segmented-reduce-argmin
  int num_segments                  = 3;
  c2h::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                 = thrust::raw_pointer_cast(d_offsets.data());
  c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<cub::KeyValuePair<int, int>> d_out(3);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::ArgMin(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::ArgMin(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  c2h::device_vector<cub::KeyValuePair<int, int>> expected{{1, 6}, {1, INT_MAX}, {2, 0}};
  // example-end segmented-reduce-argmin

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), expected.begin(), is_equal()));
}

C2H_TEST("cub::DeviceSegmentedReduce::Max works with int data elements", "[segmented_reduce][device]")
{
  // example-begin segmented-reduce-max
  int num_segments                  = 3;
  c2h::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                 = thrust::raw_pointer_cast(d_offsets.data());
  c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<int> d_out(3);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Max(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::Max(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  c2h::device_vector<int> expected{8, INT_MIN, 9};
  // example-end segmented-reduce-max

  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedReduce::ArgMax works with int data elements", "[segmented_reduce][device]")
{
  // example-begin segmented-reduce-argmax
  int num_segments                  = 3;
  c2h::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                 = thrust::raw_pointer_cast(d_offsets.data());
  c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<cub::KeyValuePair<int, int>> d_out(3);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::ArgMax(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::ArgMax(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1);

  c2h::device_vector<cub::KeyValuePair<int, int>> expected{{0, 8}, {1, INT_MIN}, {3, 9}};
  // example-end segmented-reduce-argmax

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), expected.begin(), is_equal()));
}

C2H_TEST("cub::DeviceSegmentedReduce::Reduce Fixed Segment Size works with int data elements",
         "[segmented_reduce][device]")
{
  // example-begin fixed-size-segmented-reduce-reduce
  int num_segments = 3;
  int segment_size = 2;
  c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<int> d_out(3);
  CustomMin min_op;
  int initial_value{INT_MAX};

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Reduce(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, segment_size, min_op, initial_value);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::Reduce(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, segment_size, min_op, initial_value);

  c2h::device_vector<int> expected{6, 5, 0};
  // example-end fixed-size-segmented-reduce-reduce

  REQUIRE(d_out == expected);
}

C2H_TEST("cub::DeviceSegmentedReduce::Sum Fixed Segment Size works with int data elements",
         "[segmented_reduce][device]")
{
  // example-begin fixed-size-segmented-reduce-sum
  int num_segments = 3;
  int segment_size = 2;
  c2h::device_vector<int> d_in{6, 8, 7, 5, 3, 0};
  c2h::device_vector<int> d_out(3);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, segment_size);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::Sum(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, segment_size);

  c2h::device_vector<int> d_expected{14, 12, 3};
  // example-end fixed-size-segmented-reduce-sum

  REQUIRE(d_expected == d_out);
}

C2H_TEST("cub::DeviceSegmentedReduce::Min Fixed Segment Size works with int data elements",
         "[segmented_reduce][device]")
{
  // example-begin fixed-size-segmented-reduce-min
  int num_segments = 3;
  int segment_size = 2;
  c2h::device_vector<int> d_in{6, 8, 7, 5, 3, 0};
  c2h::device_vector<int> d_out(3);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Min(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, segment_size);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::Min(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, segment_size);

  c2h::device_vector<int> d_expected{6, 5, 0};
  // example-end fixed-size-segmented-reduce-min

  REQUIRE(d_expected == d_out);
}

C2H_TEST("cub::DeviceSegmentedReduce::ArgMin Fixed Segment Size works with int data elements",
         "[segmented_reduce][device]")
{
  // example-begin fixed-size-segmented-reduce-argmin
  int num_segments = 3;
  int segment_size = 2;
  c2h::device_vector<int> d_in{6, 8, 7, 5, 3, 0};
  c2h::device_vector<cuda::std::pair<int, int>> d_out(3);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::ArgMin(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, segment_size);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::ArgMin(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, segment_size);

  c2h::host_vector<cuda::std::pair<int, int>> h_expected{{0, 6}, {1, 5}, {1, 0}};
  // example-end fixed-size-segmented-reduce-argmin

  c2h::host_vector<cuda::std::pair<int, int>> h_out(d_out);

  REQUIRE(h_expected == h_out);
}

C2H_TEST("cub::DeviceSegmentedReduce::Max Fixed Segment Size works with int data elements",
         "[segmented_reduce][device]")
{
  // example-begin fixed-size-segmented-reduce-max
  int num_segments = 3;
  int segment_size = 2;

  c2h::device_vector<int> d_in{6, 8, 7, 5, 3, 0};
  c2h::device_vector<int> d_out(3);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Max(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, segment_size);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::Max(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, segment_size);

  c2h::device_vector<int> d_expected{8, 7, 3};
  // example-end fixed-size-segmented-reduce-max

  REQUIRE(d_expected == d_out);
}

C2H_TEST("cub::DeviceSegmentedReduce::ArgMax Fixed Segment Size works with int data elements",
         "[segmented_reduce][device]")
{
  // example-begin fixed-size-segmented-reduce-argmax
  int num_segments = 3;
  int segment_size = 2;
  c2h::device_vector<int> d_in{6, 8, 7, 5, 3, 0};
  c2h::device_vector<cuda::std::pair<int, int>> d_out(3);

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::ArgMax(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, segment_size);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run reduction
  cub::DeviceSegmentedReduce::ArgMax(
    d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_segments, segment_size);

  c2h::host_vector<cuda::std::pair<int, int>> h_expected{{1, 8}, {0, 7}, {0, 3}};
  // example-end fixed-size-segmented-reduce-argmax

  c2h::host_vector<cuda::std::pair<int, int>> h_out(d_out);
  REQUIRE(h_expected == h_out);
}
