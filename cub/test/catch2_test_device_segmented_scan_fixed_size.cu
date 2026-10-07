// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_scan.cuh>

#include <thrust/tabulate.h>

#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/limits>

#include "catch2_test_device_scan.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::InclusiveSegmentedSum, device_inclusive_segmented_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::ExclusiveSegmentedSum, device_exclusive_segmented_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::InclusiveSegmentedScan, device_inclusive_segmented_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::ExclusiveSegmentedScan, device_exclusive_segmented_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::InclusiveSegmentedScanInit, device_inclusive_segmented_scan_with_init);

// %PARAM% TEST_LAUNCH lid 0:1:2

namespace
{
// Composition of affine maps x -> a * x + b, which is associative but not commutative
struct affine_t
{
  unsigned a;
  unsigned b;

  __host__ __device__ friend bool operator==(const affine_t& lhs, const affine_t& rhs)
  {
    return lhs.a == rhs.a && lhs.b == rhs.b;
  }
};

struct compose_op
{
  __host__ __device__ affine_t operator()(affine_t f, affine_t g) const
  {
    return {g.a * f.a, g.a * f.b + g.b};
  }
};

struct make_affine
{
  __host__ __device__ affine_t operator()(cuda::std::int64_t i) const
  {
    return {static_cast<unsigned>(i % 5 + 1), static_cast<unsigned>(i % 11)};
  }
};

template <typename T>
struct make_value
{
  __host__ __device__ T operator()(cuda::std::int64_t i) const
  {
    return static_cast<T>(i % 7);
  }
};

template <bool Inclusive, typename T, typename OpT, typename InitT>
c2h::host_vector<T> segmented_reference(
  const c2h::host_vector<T>& input, cuda::std::int64_t num_segments, int segment_size, OpT op, InitT init)
{
  c2h::host_vector<T> expected(input.size());
  for (cuda::std::int64_t i = 0; i < num_segments; ++i)
  {
    const auto first = input.cbegin() + i * segment_size;
    const auto out   = expected.begin() + i * segment_size;
    if constexpr (Inclusive)
    {
      compute_inclusive_scan_reference(first, first + segment_size, out, op, init);
    }
    else
    {
      compute_exclusive_scan_reference(first, first + segment_size, out, init, op);
    }
  }
  return expected;
}

// Checks inclusive sums of constant ones without materializing input or output
struct check_ones_op
{
  int segment_size;
  unsigned long long* counters; // [0] mismatches, [1] writes

  __device__ void operator()(cuda::std::int64_t i, cuda::std::int64_t value) const
  {
    atomicAdd(counters + 1, 1ull);
    if (value != i % segment_size + 1)
    {
      atomicAdd(counters, 1ull);
    }
  }
};
} // namespace

using value_types = c2h::type_list<cuda::std::uint8_t, cuda::std::int32_t, cuda::std::int64_t>;

CUB_TEST("DeviceSegmentedScan fixed-size overloads work", "[segmented][scan][fixed_size]", CUB_SMALL, value_types)
{
  using value_t = c2h::get<0, TestType>;

  // Sizes around the default tile size, in which several segments are grouped into one block
  const int segment_size  = GENERATE(1, 2, 3, 31, 64, 575, 576, 577, 1151, 1152, 1153, 5000);
  const auto num_segments = static_cast<cuda::std::int64_t>(GENERATE_COPY(1, 2, 3, 17, (1 << 16) / segment_size + 1));
  const auto num_items    = num_segments * segment_size;
  CAPTURE(c2h::type_name<value_t>(), segment_size, num_segments);

  c2h::device_vector<value_t> input(num_items, thrust::no_init);
  thrust::tabulate(input.begin(), input.end(), make_value<value_t>{});
  c2h::device_vector<value_t> output(num_items, thrust::no_init);
  const c2h::host_vector<value_t> h_input = input;

  const auto d_in  = thrust::raw_pointer_cast(input.data());
  auto d_out       = thrust::raw_pointer_cast(output.data());
  const auto plus  = cuda::std::plus<value_t>{};
  const auto max   = cuda::maximum<value_t>{};
  const auto init  = value_t{3};
  const auto empty = value_t{};

  SECTION("exclusive sum")
  {
    device_exclusive_segmented_sum(d_in, d_out, num_segments, segment_size);
    REQUIRE(segmented_reference<false>(h_input, num_segments, segment_size, plus, empty)
            == c2h::host_vector<value_t>(output));
  }

  SECTION("exclusive scan")
  {
    device_exclusive_segmented_scan(d_in, d_out, num_segments, segment_size, max, init);
    REQUIRE(
      segmented_reference<false>(h_input, num_segments, segment_size, max, init) == c2h::host_vector<value_t>(output));
  }

  SECTION("inclusive sum")
  {
    device_inclusive_segmented_sum(d_in, d_out, num_segments, segment_size);
    REQUIRE(
      segmented_reference<true>(h_input, num_segments, segment_size, plus, empty) == c2h::host_vector<value_t>(output));
  }

  SECTION("inclusive scan")
  {
    device_inclusive_segmented_scan(d_in, d_out, num_segments, segment_size, max);
    REQUIRE(
      segmented_reference<true>(h_input, num_segments, segment_size, max, empty) == c2h::host_vector<value_t>(output));
  }

  SECTION("inclusive scan with init")
  {
    device_inclusive_segmented_scan_with_init(d_in, d_out, num_segments, segment_size, plus, init);
    REQUIRE(
      segmented_reference<true>(h_input, num_segments, segment_size, plus, init) == c2h::host_vector<value_t>(output));
  }

  SECTION("in-place inclusive sum")
  {
    device_inclusive_segmented_sum(d_in, d_in, num_segments, segment_size);
    REQUIRE(
      segmented_reference<true>(h_input, num_segments, segment_size, plus, empty) == c2h::host_vector<value_t>(input));
  }
}

CUB_TEST("DeviceSegmentedScan fixed-size overloads preserve operand order", "[segmented][scan][fixed_size]", CUB_SMALL)
{
  const int segment_size  = GENERATE(1, 5, 100, 1500);
  const auto num_segments = static_cast<cuda::std::int64_t>(GENERATE(1, 7, 300));
  const auto num_items    = num_segments * segment_size;
  CAPTURE(segment_size, num_segments);

  // Fancy input iterator, to also cover non-pointer inputs
  const auto d_in = cuda::make_transform_iterator(cuda::counting_iterator<cuda::std::int64_t>{0}, make_affine{});
  const c2h::host_vector<affine_t> h_input(d_in, d_in + num_items);
  c2h::device_vector<affine_t> output(num_items, thrust::no_init);
  auto d_out = thrust::raw_pointer_cast(output.data());

  const compose_op op{};
  const affine_t init{3, 2};

  SECTION("exclusive scan")
  {
    device_exclusive_segmented_scan(d_in, d_out, num_segments, segment_size, op, init);
    REQUIRE(
      segmented_reference<false>(h_input, num_segments, segment_size, op, init) == c2h::host_vector<affine_t>(output));
  }

  SECTION("inclusive scan")
  {
    device_inclusive_segmented_scan(d_in, d_out, num_segments, segment_size, op);
    REQUIRE(segmented_reference<true>(h_input, num_segments, segment_size, op, affine_t{1, 0})
            == c2h::host_vector<affine_t>(output));
  }

  SECTION("inclusive scan with init")
  {
    device_inclusive_segmented_scan_with_init(d_in, d_out, num_segments, segment_size, op, init);
    REQUIRE(
      segmented_reference<true>(h_input, num_segments, segment_size, op, init) == c2h::host_vector<affine_t>(output));
  }
}

CUB_TEST("DeviceSegmentedScan fixed-size overloads handle more than INT_MAX items or segments",
         "[segmented][scan][fixed_size]",
         CUB_SMALL)
{
  // {num_segments, segment_size}: one block per segment with 64-bit item offsets, and several kernel launches
  const auto [num_segments, segment_size] = GENERATE(
    table<cuda::std::int64_t, int>({{3, 1 << 30}, {(cuda::std::int64_t{1} << 31) + 5, 1}, {(1 << 30) + 3, 3}}));
  CAPTURE(num_segments, segment_size);

  c2h::device_vector<unsigned long long> counters(2, 0);
  const auto d_in  = cuda::constant_iterator<cuda::std::int64_t>{1};
  const auto d_out = cuda::tabulate_output_iterator{
    check_ones_op{segment_size, thrust::raw_pointer_cast(counters.data())}, cuda::std::int64_t{0}};

  device_inclusive_segmented_sum(d_in, d_out, num_segments, segment_size);

  const c2h::host_vector<unsigned long long> h_counters = counters;
  REQUIRE(h_counters[0] == 0);
  REQUIRE(h_counters[1] == static_cast<unsigned long long>(num_segments * segment_size));
}

CUB_TEST("DeviceSegmentedScan fixed-size overloads validate sizes", "[segmented][scan][fixed_size]", CUB_SMALL)
{
  int* ptr                  = nullptr;
  size_t temp_storage_bytes = 0;
  cuda::std::uint8_t storage{};

  using cub::DeviceSegmentedScan;

  // empty problems request storage, and then succeed without launching
  REQUIRE(DeviceSegmentedScan::InclusiveSegmentedSum(nullptr, temp_storage_bytes, ptr, ptr, 0, 5) == cudaSuccess);
  REQUIRE(temp_storage_bytes > 0);
  REQUIRE(DeviceSegmentedScan::InclusiveSegmentedSum(&storage, temp_storage_bytes, ptr, ptr, 0, 5) == cudaSuccess);
  REQUIRE(DeviceSegmentedScan::InclusiveSegmentedSum(&storage, temp_storage_bytes, ptr, ptr, 5, 0) == cudaSuccess);

  REQUIRE(
    DeviceSegmentedScan::InclusiveSegmentedSum(nullptr, temp_storage_bytes, ptr, ptr, -1, 5) == cudaErrorInvalidValue);
  REQUIRE(
    DeviceSegmentedScan::InclusiveSegmentedSum(nullptr, temp_storage_bytes, ptr, ptr, 5, -1) == cudaErrorInvalidValue);
  REQUIRE(DeviceSegmentedScan::InclusiveSegmentedSum(
            nullptr, temp_storage_bytes, ptr, ptr, cuda::std::numeric_limits<cuda::std::int64_t>::max(), 2)
          == cudaErrorInvalidValue);
  REQUIRE(DeviceSegmentedScan::InclusiveSegmentedSum(ptr, ptr, -1, 5) == cudaErrorInvalidValue);
}

CUB_TEST("DeviceSegmentedScan fixed-size and offset overloads resolve unambiguously",
         "[segmented][scan][fixed_size]",
         CUB_SMALL)
{
  c2h::device_vector<int> in(4, 1);
  c2h::device_vector<int> out(4);
  c2h::device_vector<int> offsets{0, 4};
  size_t temp_storage_bytes = 0;

  // segment size passed as size_t, which is a better match for the env overloads unless they are constrained
  const size_t segment_size = 4;
  REQUIRE(cub::DeviceSegmentedScan::InclusiveSegmentedSum(
            nullptr, temp_storage_bytes, in.begin(), out.begin(), 1, segment_size)
          == cudaSuccess);
  REQUIRE(cub::DeviceSegmentedScan::ExclusiveSegmentedScan(
            nullptr, temp_storage_bytes, in.begin(), out.begin(), 1, segment_size, cuda::std::plus<>{}, 0)
          == cudaSuccess);
  REQUIRE(cub::DeviceSegmentedScan::InclusiveSegmentedScan(
            nullptr, temp_storage_bytes, in.begin(), out.begin(), 1, segment_size, cuda::std::plus<>{})
          == cudaSuccess);
  REQUIRE(cub::DeviceSegmentedScan::InclusiveSegmentedSum(
            nullptr, temp_storage_bytes, in.begin(), out.begin(), offsets.begin(), offsets.begin() + 1, 1)
          == cudaSuccess);
}
