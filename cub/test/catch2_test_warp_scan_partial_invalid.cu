// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_scan.cuh>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/limits>

#include "catch2_test_warp_scan_partial_helper.cuh"
#include "thread_reduce/catch2_test_thread_reduce_helper.cuh"
#include <c2h/catch2_test_helper.h>

using invalid_types        = c2h::type_list<segment>;
using logical_warp_threads = c2h::enum_type_list<int, 32, 16, 9, 2>;
using modes                = c2h::enum_type_list<scan_mode, scan_mode::exclusive, scan_mode::inclusive>;

template <scan_mode Mode>
struct merge_op_t
{
  bool* error_flag_ptr;
  template <class WarpScanT>
  __device__ void operator()(WarpScanT& scan, segment& thread_data, int valid_items) const
  {
    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartial(thread_data, thread_data, merge_segments_op{error_flag_ptr}, valid_items);
    }
    else
    {
      scan.InclusiveScanPartial(thread_data, thread_data, merge_segments_op{error_flag_ptr}, valid_items);
    }
  }
};

template <scan_mode Mode>
struct merge_aggregate_op_t
{
  int m_target_thread_id;
  segment* m_d_warp_aggregate;
  bool* error_flag_ptr;

  template <int LogicalWarpThreads>
  __device__ void
  operator()(cub::WarpScan<segment, LogicalWarpThreads>& scan, segment& thread_data, int valid_items) const
  {
    segment warp_aggregate{};

    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartial(
        thread_data, thread_data, merge_segments_op{error_flag_ptr}, valid_items, warp_aggregate);
    }
    else
    {
      scan.InclusiveScanPartial(
        thread_data, thread_data, merge_segments_op{error_flag_ptr}, valid_items, warp_aggregate);
    }

    const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

    if (tid % LogicalWarpThreads == m_target_thread_id)
    {
      m_d_warp_aggregate[tid / LogicalWarpThreads] = warp_aggregate;
    }
  }
};

template <scan_mode Mode>
struct merge_init_value_op_t
{
  segment initial_value;
  bool* error_flag_ptr;

  template <class WarpScanT>
  __device__ void operator()(WarpScanT& scan, segment& thread_data, int valid_items) const
  {
    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartial(thread_data, thread_data, initial_value, merge_segments_op{error_flag_ptr}, valid_items);
    }
    else
    {
      scan.InclusiveScanPartial(thread_data, thread_data, initial_value, merge_segments_op{error_flag_ptr}, valid_items);
    }
  }
};

template <scan_mode Mode>
struct merge_init_value_aggregate_op_t
{
  int m_target_thread_id;
  segment initial_value;
  segment* m_d_warp_aggregate;
  bool* error_flag_ptr;

  template <int LogicalWarpThreads>
  __device__ void
  operator()(cub::WarpScan<segment, LogicalWarpThreads>& scan, segment& thread_data, int valid_items) const
  {
    segment warp_aggregate{};

    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartial(
        thread_data, thread_data, initial_value, merge_segments_op{error_flag_ptr}, valid_items, warp_aggregate);
    }
    else
    {
      scan.InclusiveScanPartial(
        thread_data, thread_data, initial_value, merge_segments_op{error_flag_ptr}, valid_items, warp_aggregate);
    }

    const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

    if (tid % LogicalWarpThreads == m_target_thread_id)
    {
      m_d_warp_aggregate[tid / LogicalWarpThreads] = warp_aggregate;
    }
  }
};

struct merge_scan_op_t
{
  bool* error_flag_ptr;

  template <class WarpScanT>
  __device__ void operator()(
    WarpScanT& scan, segment& thread_data, segment& inclusive_output, segment& exclusive_output, int valid_items) const
  {
    scan.ScanPartial(thread_data, inclusive_output, exclusive_output, merge_segments_op{error_flag_ptr}, valid_items);
  }
};

struct merge_init_value_scan_op_t
{
  segment initial_value;
  bool* error_flag_ptr;

  template <class WarpScanT>
  __device__ void operator()(
    WarpScanT& scan, segment& thread_data, segment& inclusive_output, segment& exclusive_output, int valid_items) const
  {
    scan.ScanPartial(
      thread_data, inclusive_output, exclusive_output, initial_value, merge_segments_op{error_flag_ptr}, valid_items);
  }
};

C2H_TEST(
  "Partial warp scan does not apply op to invalid elements", "[scan][warp]", invalid_types, logical_warp_threads, modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  const int valid_items = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, params::logical_warp_threads))),
    take(1, random(params::logical_warp_threads + 2, cuda::std::numeric_limits<int>::max())),
    take(1, random(cuda::std::numeric_limits<int>::min(), -2)),
    values({-1, 0, 1, params::logical_warp_threads, params::logical_warp_threads + 1}));
  const int bounded_valid_items = cuda::std::clamp(valid_items, 0, params::logical_warp_threads);
  CAPTURE(valid_items, params::mode, params::logical_warp_threads, c2h::type_name<type>());
  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  const auto in_it = cuda::make_transform_iterator(
    thrust::make_zip_iterator(cuda::counting_iterator<segment::offset_t>{1},
                              cuda::counting_iterator<segment::offset_t>{2}),
    tuple_to_segment_op{});
  for (size_t i = 0; i < params::tile_size; i += params::logical_warp_threads)
  {
    thrust::copy(in_it, in_it + bounded_valid_items, d_in.begin() + i);
  }

  c2h::device_vector<bool> error_flag(1);
  warp_scan<params::logical_warp_threads, params::total_warps>(
    d_in, d_out, merge_op_t<params::mode>{thrust::raw_pointer_cast(error_flag.data())}, valid_items);
  REQUIRE(false == error_flag.front());
  c2h::host_vector<type> h_out = d_in;

  compute_host_reference(
    params::mode, h_out, params::logical_warp_threads, merge_segments_op{}, valid_items, segment{1, 1});

  // From the documentation -
  // Computes an exclusive prefix scan using the specified binary scan functor across the calling warp. Because no
  // initial value is supplied, the output computed for warp-lane0 is undefined.

  // When comparing device output, the corresponding undefined data points need to be fixed

  if constexpr (params::mode == scan_mode::exclusive)
  {
    for (size_t i = 0; i < h_out.size(); i += params::logical_warp_threads)
    {
      d_out[i] = h_out[i];
    }
  }
  REQUIRE(h_out == d_out);
}

C2H_TEST("Partial warp scan does not apply op to invalid elements and returns valid warp aggregate",
         "[scan][warp]",
         invalid_types,
         logical_warp_threads,
         modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  const int valid_items = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, params::logical_warp_threads))),
    take(1, random(params::logical_warp_threads + 2, cuda::std::numeric_limits<int>::max())),
    take(1, random(cuda::std::numeric_limits<int>::min(), -2)),
    values({-1, 0, 1, params::logical_warp_threads, params::logical_warp_threads + 1}));
  const int bounded_valid_items = cuda::std::clamp(valid_items, 0, params::logical_warp_threads);
  CAPTURE(valid_items, params::mode, params::logical_warp_threads, c2h::type_name<type>());
  c2h::device_vector<type> d_warp_aggregates(params::total_warps);
  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  const auto in_it = cuda::make_transform_iterator(
    thrust::make_zip_iterator(cuda::counting_iterator<segment::offset_t>{1},
                              cuda::counting_iterator<segment::offset_t>{2}),
    tuple_to_segment_op{});
  for (size_t i = 0; i < params::tile_size; i += params::logical_warp_threads)
  {
    thrust::copy(in_it, in_it + bounded_valid_items, d_in.begin() + i);
  }

  const int target_thread_id = GENERATE_COPY(take(2, random(0, params::logical_warp_threads - 1)));

  c2h::device_vector<bool> error_flag(1);
  warp_scan<params::logical_warp_threads, params::total_warps>(
    d_in,
    d_out,
    merge_aggregate_op_t<params::mode>{
      target_thread_id, thrust::raw_pointer_cast(d_warp_aggregates.data()), thrust::raw_pointer_cast(error_flag.data())},
    valid_items);
  REQUIRE(false == error_flag.front());

  c2h::host_vector<type> h_out = d_in;

  auto h_warp_aggregates = compute_host_reference(
    params::mode, h_out, params::logical_warp_threads, merge_segments_op{}, valid_items, segment{1, 1});

  // From the documentation -
  // Computes an exclusive prefix scan using the specified binary scan functor across the calling warp. Because no
  // initial value is supplied, the output computed for warp-lane0 is undefined.

  // When comparing device output, the corresponding undefined data points need to be fixed

  if constexpr (params::mode == scan_mode::exclusive)
  {
    for (size_t i = 0; i < h_out.size(); i += params::logical_warp_threads)
    {
      d_out[i] = h_out[i];
    }
  }
  REQUIRE(h_out == d_out);
  if (valid_items > 0)
  {
    REQUIRE(h_warp_aggregates == d_warp_aggregates);
  }
}

C2H_TEST("Partial warp scan does not apply op to invalid elements and works with initial value",
         "[scan][warp]",
         invalid_types,
         logical_warp_threads,
         modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  const int valid_items = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, params::logical_warp_threads))),
    take(1, random(params::logical_warp_threads + 2, cuda::std::numeric_limits<int>::max())),
    take(1, random(cuda::std::numeric_limits<int>::min(), -2)),
    values({-1, 0, 1, params::logical_warp_threads, params::logical_warp_threads + 1}));
  const int bounded_valid_items = cuda::std::clamp(valid_items, 0, params::logical_warp_threads);
  CAPTURE(valid_items, params::mode, params::logical_warp_threads, c2h::type_name<type>());
  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  const auto in_it = cuda::make_transform_iterator(
    thrust::make_zip_iterator(cuda::counting_iterator<segment::offset_t>{1},
                              cuda::counting_iterator<segment::offset_t>{2}),
    tuple_to_segment_op{});
  for (size_t i = 0; i < params::tile_size; i += params::logical_warp_threads)
  {
    thrust::copy(in_it, in_it + bounded_valid_items, d_in.begin() + i);
  }

  const type initial_value = segment{0, 1};

  c2h::device_vector<bool> error_flag(1);
  warp_scan<params::logical_warp_threads, params::total_warps>(
    d_in,
    d_out,
    merge_init_value_op_t<params::mode>{initial_value, thrust::raw_pointer_cast(error_flag.data())},
    valid_items);
  REQUIRE(false == error_flag.front());

  c2h::host_vector<type> h_out = d_in;

  compute_host_reference(
    params::mode, h_out, params::logical_warp_threads, merge_segments_op{}, valid_items, initial_value);

  REQUIRE(h_out == d_out);
}

C2H_TEST("Partial warp scan with initial value does not apply op to invalid elements and returns valid warp aggregate",
         "[scan][warp]",
         invalid_types,
         logical_warp_threads,
         modes)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  const int valid_items = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, params::logical_warp_threads))),
    take(1, random(params::logical_warp_threads + 2, cuda::std::numeric_limits<int>::max())),
    take(1, random(cuda::std::numeric_limits<int>::min(), -2)),
    values({-1, 0, 1, params::logical_warp_threads, params::logical_warp_threads + 1}));
  const int bounded_valid_items = cuda::std::clamp(valid_items, 0, params::logical_warp_threads);
  CAPTURE(valid_items, params::mode, params::logical_warp_threads, c2h::type_name<type>());
  c2h::device_vector<type> d_warp_aggregates(params::total_warps);
  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  const auto in_it = cuda::make_transform_iterator(
    thrust::make_zip_iterator(cuda::counting_iterator<segment::offset_t>{1},
                              cuda::counting_iterator<segment::offset_t>{2}),
    tuple_to_segment_op{});
  for (size_t i = 0; i < params::tile_size; i += params::logical_warp_threads)
  {
    thrust::copy(in_it, in_it + bounded_valid_items, d_in.begin() + i);
  }

  const int target_thread_id = GENERATE_COPY(take(2, random(0, params::logical_warp_threads - 1)));
  const type initial_value   = segment{0, 1};

  c2h::device_vector<bool> error_flag(1);
  warp_scan<params::logical_warp_threads, params::total_warps>(
    d_in,
    d_out,
    merge_init_value_aggregate_op_t<params::mode>{
      target_thread_id,
      initial_value,
      thrust::raw_pointer_cast(d_warp_aggregates.data()),
      thrust::raw_pointer_cast(error_flag.data())},
    valid_items);
  REQUIRE(false == error_flag.front());

  c2h::host_vector<type> h_out = d_in;

  auto h_warp_aggregates = compute_host_reference(
    params::mode, h_out, params::logical_warp_threads, merge_segments_op{}, valid_items, initial_value);

  REQUIRE(h_out == d_out);
  if (valid_items > 0)
  {
    REQUIRE(h_warp_aggregates == d_warp_aggregates);
  }
}

C2H_TEST("Partial warp combination scan does not apply op to invalid elements", "[scan][warp]", logical_warp_threads)
{
  constexpr int logical_warp_threads = c2h::get<0, TestType>();
  constexpr int total_warps          = total_warps_t<logical_warp_threads>::value();
  constexpr int tile_size            = logical_warp_threads * total_warps;
  using type                         = segment;

  const int valid_items = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, logical_warp_threads))),
    take(1, random(logical_warp_threads + 2, cuda::std::numeric_limits<int>::max())),
    take(1, random(cuda::std::numeric_limits<int>::min(), -2)),
    values({-1, 0, 1, logical_warp_threads, logical_warp_threads + 1}));
  const int bounded_valid_items = cuda::std::clamp(valid_items, 0, logical_warp_threads);
  CAPTURE(valid_items, logical_warp_threads, c2h::type_name<type>());
  c2h::device_vector<type> d_inclusive_out(tile_size);
  c2h::device_vector<type> d_exclusive_out(tile_size);
  c2h::device_vector<type> d_in(tile_size);
  const auto in_it = cuda::make_transform_iterator(
    thrust::make_zip_iterator(cuda::counting_iterator<segment::offset_t>{1},
                              cuda::counting_iterator<segment::offset_t>{2}),
    tuple_to_segment_op{});
  for (size_t i = 0; i < tile_size; i += logical_warp_threads)
  {
    thrust::copy(in_it, in_it + bounded_valid_items, d_in.begin() + i);
  }

  c2h::device_vector<bool> error_flag(1);
  warp_combine_scan<logical_warp_threads, total_warps>(
    d_in,
    d_inclusive_out,
    d_exclusive_out,
    merge_scan_op_t{thrust::raw_pointer_cast(error_flag.data())},
    valid_items,
    segment{});
  REQUIRE(false == error_flag.front());

  c2h::host_vector<type> h_exclusive_out = d_in;
  c2h::host_vector<type> h_inclusive_out = d_in;

  compute_host_reference(
    scan_mode::exclusive, h_exclusive_out, logical_warp_threads, merge_segments_op{}, valid_items, segment{1, 1});

  compute_host_reference(
    scan_mode::inclusive, h_inclusive_out, logical_warp_threads, merge_segments_op{}, valid_items, segment{1, 1});

  // According to WarpScan::Scan documentation -
  // Because no initial value is supplied, the exclusive_output computed for warp-lane0 is undefined.

  // When comparing device output, the corresponding undefined data points need to be fixed

  for (size_t i = 0; i < h_exclusive_out.size(); i += logical_warp_threads)
  {
    d_exclusive_out[i] = h_exclusive_out[i];
  }

  REQUIRE(h_inclusive_out == d_inclusive_out);
  REQUIRE(h_exclusive_out == d_exclusive_out);
}

C2H_TEST("Partial warp combination custom scan does not apply op to invalid elements and works with initial value",
         "[scan][warp]",
         logical_warp_threads)
{
  constexpr int logical_warp_threads = c2h::get<0, TestType>();
  constexpr int total_warps          = total_warps_t<logical_warp_threads>::value();
  constexpr int tile_size            = logical_warp_threads * total_warps;
  using type                         = segment;

  const int valid_items = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, logical_warp_threads))),
    take(1, random(logical_warp_threads + 2, cuda::std::numeric_limits<int>::max())),
    take(1, random(cuda::std::numeric_limits<int>::min(), -2)),
    values({-1, 0, 1, logical_warp_threads, logical_warp_threads + 1}));
  const int bounded_valid_items = cuda::std::clamp(valid_items, 0, logical_warp_threads);
  CAPTURE(valid_items, logical_warp_threads, c2h::type_name<type>());
  c2h::device_vector<type> d_inclusive_out(total_warps * logical_warp_threads);
  c2h::device_vector<type> d_exclusive_out(total_warps * logical_warp_threads);
  c2h::device_vector<type> d_in(total_warps * logical_warp_threads);
  const auto in_it = cuda::make_transform_iterator(
    thrust::make_zip_iterator(cuda::counting_iterator<segment::offset_t>{1},
                              cuda::counting_iterator<segment::offset_t>{2}),
    tuple_to_segment_op{});
  for (size_t i = 0; i < tile_size; i += logical_warp_threads)
  {
    thrust::copy(in_it, in_it + bounded_valid_items, d_in.begin() + i);
  }

  const type initial_value = segment{0, 1};

  c2h::device_vector<bool> error_flag(1);
  warp_combine_scan<logical_warp_threads, total_warps>(
    d_in,
    d_inclusive_out,
    d_exclusive_out,
    merge_init_value_scan_op_t{initial_value, thrust::raw_pointer_cast(error_flag.data())},
    valid_items,
    segment{});
  REQUIRE(false == error_flag.front());

  c2h::host_vector<type> h_exclusive_out = d_in;
  c2h::host_vector<type> h_inclusive_out = d_in;

  compute_host_reference(
    scan_mode::exclusive, h_exclusive_out, logical_warp_threads, merge_segments_op{}, valid_items, initial_value);

  compute_host_reference(
    scan_mode::inclusive, h_inclusive_out, logical_warp_threads, merge_segments_op{}, valid_items, initial_value);

  REQUIRE(h_inclusive_out == d_inclusive_out);
  REQUIRE(h_exclusive_out == d_exclusive_out);
}
