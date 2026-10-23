// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_scan.cuh>

#include <thrust/equal.h>
#include <thrust/fill.h>

#include <cuda/atomic>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "catch2_large_problem_helper.cuh"
#include "catch2_test_device_segmented_scan_schedules.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedScan::InclusiveSegmentedSum, device_segmented_inclusive_sum);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedScan::ExclusiveSegmentedSum, device_segmented_exclusive_sum);

// %PARAM% TEST_LAUNCH lid 0:1:2

namespace
{
using item_t   = cuda::std::uint8_t;
using offset_t = cuda::std::int32_t;

// Every input item is 1, so the inclusive sum at position k of a segment is k + 1, truncated to item_t.
struct expected_ones_sum
{
  cuda::std::int64_t segment_size;

  __host__ __device__ item_t operator()(cuda::std::int64_t index) const
  {
    return static_cast<item_t>(index % segment_size + 1);
  }
};

// Segments 0, num_segments / 2 and num_segments - 1 hold 1000 items each, back to back; all others are empty.
template <bool End>
struct sparse_offset
{
  cuda::std::int64_t num_segments;

  __host__ __device__ offset_t operator()(cuda::std::int64_t segment) const
  {
    const offset_t extent = End ? 1000 : 0;
    if (segment == 0)
    {
      return extent;
    }
    if (segment == num_segments / 2)
    {
      return 1000 + extent;
    }
    if (segment == num_segments - 1)
    {
      return 2000 + extent;
    }
    return 0;
  }
};

constexpr cuda::std::int64_t total_past_max = cuda::std::int64_t{1} << 31;
constexpr offset_t long_segment_items       = static_cast<offset_t>(total_past_max - 3);

// The first four segments hold 2^31 - 3, 1, 1 and 1 items, all reading from input 0 and writing back to back so that
// their outputs cover [0, 2^31). All other segments are empty.
struct one_block_end_offset
{
  __host__ __device__ offset_t operator()(cuda::std::int64_t segment) const
  {
    return segment == 0 ? long_segment_items : (segment < 4 ? 1 : 0);
  }
};

struct one_block_output_offset
{
  __host__ __device__ offset_t operator()(cuda::std::int64_t segment) const
  {
    return segment == 0 || segment > 3 ? 0 : long_segment_items + static_cast<offset_t>(segment - 1);
  }
};

struct one_block_expected
{
  __host__ __device__ item_t operator()(cuda::std::int64_t index) const
  {
    return index < long_segment_items ? static_cast<item_t>(index + 1) : item_t{1};
  }
};

__host__ __device__ void count_stray_write(unsigned long long* stray_writes)
{
  cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>{*stray_writes}.fetch_add(
    1, cuda::std::memory_order_relaxed);
}

// Passes a write at output position first_position + k, for k in [0, num_positions), to check_op as position k, and
// counts every other write in *stray_writes.
template <typename CheckOpT>
struct bounded_write_op
{
  CheckOpT check_op;
  cuda::std::int64_t first_position;
  cuda::std::int64_t num_positions;
  unsigned long long* stray_writes;

  template <typename T>
  __host__ __device__ void operator()(cuda::std::int64_t index, T value)
  {
    const cuda::std::int64_t position = index - first_position;
    if (position >= 0 && position < num_positions)
    {
      check_op(position, value);
    }
    else
    {
      count_stray_write(stray_writes);
    }
  }
};

template <typename CheckOpT>
bounded_write_op<CheckOpT> make_bounded_write_op(
  CheckOpT check_op,
  cuda::std::int64_t first_position,
  cuda::std::int64_t num_positions,
  unsigned long long* stray_writes)
{
  return bounded_write_op<CheckOpT>{check_op, first_position, num_positions, stray_writes};
}

// Every input item is 1 and the output segments are back to back from position 0; the last segment holds the
// remainder.
struct expected_back_to_back_ones_sum
{
  cuda::std::int64_t segment_items;
  cuda::std::int64_t last_segment;

  __host__ __device__ int operator()(cuda::std::int64_t position) const
  {
    const cuda::std::int64_t segment = cuda::std::min(position / segment_items, last_segment);
    return static_cast<int>(position - segment * segment_items + 1);
  }
};

// Stores a write at output position first + k, for k in {0, 1}, in slots[k], and at second + k, for k in {0, 1, 2}, in
// slots[2 + k], and counts every other write in *stray_writes.
struct two_window_recorder
{
  cuda::std::int64_t first;
  cuda::std::int64_t second;
  int* slots;
  unsigned long long* stray_writes;

  template <typename T>
  __host__ __device__ void operator()(cuda::std::int64_t index, T value) const
  {
    if (index - first >= 0 && index - first < 2)
    {
      slots[index - first] = static_cast<int>(value);
    }
    else if (index - second >= 0 && index - second < 3)
    {
      slots[2 + index - second] = static_cast<int>(value);
    }
    else
    {
      count_stray_write(stray_writes);
    }
  }
};

// The first three segments, of two items at inputs [0, 2), two at [2, 4) and one at [4, 5), write to output positions
// OffsetT max - 1, OffsetT min and OffsetT min + 2; the first two are adjacent modulo 2^n but 2^n - 2 positions apart,
// and any further segment is empty. The output iterator records writes with 64-bit positions, so a store that runs on
// from max - 1 past max lands outside the segments and is counted instead of written. The default path scans each
// segment in its own block; with num_items alone, the segments share one block; with load balancing, one block scans
// the range from the head of its last segment separately, so the third segment keeps the first two in one tile.
template <typename OffsetT>
void check_outputs_either_side_of_the_wrap(
  const c2h::device_vector<OffsetT>& begins,
  const c2h::device_vector<OffsetT>& ends,
  const c2h::device_vector<OffsetT>& out_begin)
{
  constexpr cuda::std::int64_t first       = cuda::std::int64_t{cuda::std::numeric_limits<OffsetT>::max()} - 1;
  constexpr cuda::std::int64_t second      = cuda::std::int64_t{cuda::std::numeric_limits<OffsetT>::min()};
  constexpr cuda::std::int64_t total_items = 5;
  const cuda::std::int64_t num_segments    = static_cast<cuda::std::int64_t>(begins.size());
  const OffsetT* d_begins                  = thrust::raw_pointer_cast(begins.data());
  const OffsetT* d_ends                    = thrust::raw_pointer_cast(ends.data());
  const OffsetT* d_out_begin               = thrust::raw_pointer_cast(out_begin.data());
  const auto d_in                          = cuda::counting_iterator(int{1});

  c2h::device_vector<int> slots(5);
  c2h::device_vector<unsigned long long> stray_writes(1);
  const auto d_out = cuda::make_tabulate_output_iterator(two_window_recorder{
    first, second, thrust::raw_pointer_cast(slots.data()), thrust::raw_pointer_cast(stray_writes.data())});

  // gcc 7 ICEs when a generic lambda captures num_segments implicitly
  const auto run = [&, num_segments](const auto& schedule_env) {
    thrust::fill(c2h::device_policy, slots.begin(), slots.end(), -1);
    thrust::fill(c2h::device_policy, stray_writes.begin(), stray_writes.end(), 0ull);
    const size_t bytes = schedule_allocation_size<cub::ForceInclusive::No, false>(
      schedule_env, d_in, d_out, num_segments, d_begins, d_ends, d_out_begin, cuda::std::plus<>{}, cub::NullType{});
    device_segmented_inclusive_sum(
      d_in,
      d_out,
      d_begins,
      d_ends,
      d_out_begin,
      num_segments,
      cuda::std::execution::env{expected_allocation_size(bytes), schedule_env});

    const c2h::host_vector<int> expected = {1, 3, 3, 7, 5};
    REQUIRE(stray_writes[0] == 0ull);
    REQUIRE(c2h::host_vector<int>(slots) == expected);
  };

  SECTION("default")
  {
    run(cuda::std::execution::env{});
  }

  SECTION("with num_items")
  {
    run(cuda::std::execution::env{cub::segmented_scan_num_items(total_items)});
  }

  SECTION("with load balancing")
  {
    run(cuda::std::execution::env{cub::segmented_scan_load_balancing});
  }

  SECTION("with load balancing and num_items")
  {
    run(cuda::std::execution::env{cub::segmented_scan_num_items(total_items), cub::segmented_scan_load_balancing});
  }
}

// Passes a write at output position first + k, for k in [0, first_items), to check_op as position k and a write at
// second + k, for k in [0, later_items), as position first_items + k, and counts every other write in *stray_writes.
template <typename CheckOpT>
struct wrapped_write_op
{
  CheckOpT check_op;
  cuda::std::int64_t first;
  cuda::std::int64_t first_items;
  cuda::std::int64_t second;
  cuda::std::int64_t later_items;
  unsigned long long* stray_writes;

  template <typename T>
  __host__ __device__ void operator()(cuda::std::int64_t index, T value)
  {
    if (index - first >= 0 && index - first < first_items)
    {
      check_op(index - first, value);
    }
    else if (index - second >= 0 && index - second < later_items)
    {
      check_op(first_items + index - second, value);
    }
    else
    {
      count_stray_write(stray_writes);
    }
  }
};

template <typename CheckOpT>
wrapped_write_op<CheckOpT> make_wrapped_write_op(
  CheckOpT check_op,
  cuda::std::int64_t first,
  cuda::std::int64_t first_items,
  cuda::std::int64_t second,
  cuda::std::int64_t later_items,
  unsigned long long* stray_writes)
{
  return wrapped_write_op<CheckOpT>{check_op, first, first_items, second, later_items, stray_writes};
}

// Every input item is 1 and the segments hold first_items, second_items and the remaining positions in that order, so
// the sum at position k of a segment is k + 1 inclusively and k exclusively.
struct expected_three_segment_ones_sum
{
  cuda::std::int64_t first_items;
  cuda::std::int64_t second_items;
  bool inclusive;

  __host__ __device__ cuda::std::uint32_t operator()(cuda::std::int64_t position) const
  {
    const cuda::std::int64_t segment_begin =
      position < first_items ? 0 : (position < first_items + second_items ? first_items : first_items + second_items);
    return static_cast<cuda::std::uint32_t>(position - segment_begin + (inclusive ? 1 : 0));
  }
};

// Segment 0 holds first_items items and writes up to the offset maximum; segments 1 and 2 hold second_items and
// third_items and write back to back from the offset minimum, so every output segment follows the one before it modulo
// 2^n. With gapped, an empty fourth segment writes elsewhere, so not every one does, but a tile holding segments 0 and
// 1 still has output deltas that are equal modulo 2^n. The inputs are back to back. The output iterator records writes
// with 64-bit positions, so a store that runs on past the maximum is counted instead of written. Each schedule scans
// inclusively and exclusively; with_block_paths adds the default path and num_items alone.
template <typename OffsetT>
void check_scans_across_the_wrap(
  cuda::std::int64_t first_items,
  cuda::std::int64_t second_items,
  cuda::std::int64_t third_items,
  bool gapped,
  bool with_block_paths)
{
  constexpr cuda::std::int64_t offset_max  = cuda::std::int64_t{cuda::std::numeric_limits<OffsetT>::max()};
  constexpr cuda::std::int64_t offset_min  = cuda::std::int64_t{cuda::std::numeric_limits<OffsetT>::min()};
  const cuda::std::int64_t total_items     = first_items + second_items + third_items;
  const cuda::std::int64_t first           = offset_max - first_items + 1;
  const cuda::std::int64_t input_begins[]  = {0, first_items, first_items + second_items, total_items};
  const cuda::std::int64_t output_begins[] = {first, offset_min, offset_min + second_items, offset_max / 2};
  const cuda::std::int64_t num_segments    = gapped ? 4 : 3;
  c2h::host_vector<OffsetT> h_begins(static_cast<size_t>(num_segments));
  c2h::host_vector<OffsetT> h_ends(static_cast<size_t>(num_segments));
  c2h::host_vector<OffsetT> h_out_begin(static_cast<size_t>(num_segments));
  for (cuda::std::int64_t segment = 0; segment < num_segments; ++segment)
  {
    h_begins[segment]    = static_cast<OffsetT>(input_begins[segment]);
    h_ends[segment]      = static_cast<OffsetT>(input_begins[(cuda::std::min) (segment + 1, cuda::std::int64_t{3})]);
    h_out_begin[segment] = static_cast<OffsetT>(output_begins[segment]);
  }
  const c2h::device_vector<OffsetT> begins    = h_begins;
  const c2h::device_vector<OffsetT> ends      = h_ends;
  const c2h::device_vector<OffsetT> out_begin = h_out_begin;
  const OffsetT* d_begins                     = thrust::raw_pointer_cast(begins.data());
  const OffsetT* d_ends                       = thrust::raw_pointer_cast(ends.data());
  const OffsetT* d_out_begin                  = thrust::raw_pointer_cast(out_begin.data());
  const auto d_in                             = cuda::constant_iterator<cuda::std::uint32_t>(cuda::std::uint32_t{1});

  c2h::device_vector<unsigned long long> stray_writes(1);

  // gcc 7 ICEs when a generic lambda captures num_segments implicitly
  const auto run = [&, num_segments, total_items, first_items, second_items, first](const auto& schedule_env) {
    for (const bool inclusive : {true, false})
    {
      detail::large_problem_test_helper helper(static_cast<std::size_t>(total_items));
      const auto expected = cuda::transform_iterator(
        cuda::counting_iterator(cuda::std::int64_t{0}),
        expected_three_segment_ones_sum{first_items, second_items, inclusive});
      const auto d_out = cuda::make_tabulate_output_iterator(make_wrapped_write_op(
        detail::make_checking_write_op(expected, thrust::raw_pointer_cast(helper.correctness_flags.data())),
        first,
        first_items,
        offset_min,
        total_items - first_items,
        thrust::raw_pointer_cast(stray_writes.data())));
      thrust::fill(c2h::device_policy, stray_writes.begin(), stray_writes.end(), 0ull);
      if (inclusive)
      {
        const size_t bytes = schedule_allocation_size<cub::ForceInclusive::No, false>(
          schedule_env, d_in, d_out, num_segments, d_begins, d_ends, d_out_begin, cuda::std::plus<>{}, cub::NullType{});
        device_segmented_inclusive_sum(
          d_in,
          d_out,
          d_begins,
          d_ends,
          d_out_begin,
          num_segments,
          cuda::std::execution::env{expected_allocation_size(bytes), schedule_env});
      }
      else
      {
        const size_t bytes = schedule_allocation_size<cub::ForceInclusive::No, false>(
          schedule_env,
          d_in,
          d_out,
          num_segments,
          d_begins,
          d_ends,
          d_out_begin,
          cuda::std::plus<>{},
          cub::detail::InputValue<cuda::std::uint32_t>{cuda::std::uint32_t{0}});
        device_segmented_exclusive_sum(
          d_in,
          d_out,
          d_begins,
          d_ends,
          d_out_begin,
          num_segments,
          cuda::std::execution::env{expected_allocation_size(bytes), schedule_env});
      }
      INFO("inclusive: " << inclusive);
      REQUIRE(stray_writes[0] == 0ull);
      helper.check_all_results_correct();
    }
  };

  if (with_block_paths)
  {
    SECTION("default")
    {
      run(cuda::std::execution::env{});
    }

    SECTION("with num_items")
    {
      run(cuda::std::execution::env{cub::segmented_scan_num_items(total_items)});
    }
  }

  SECTION("with load balancing")
  {
    run(cuda::std::execution::env{cub::segmented_scan_load_balancing});
  }

  SECTION("with load balancing and num_items")
  {
    run(cuda::std::execution::env{cub::segmented_scan_num_items(total_items), cub::segmented_scan_load_balancing});
  }
}
} // namespace

// The output segments are disjoint and cover every position from 0 to the offset maximum, one more position than the
// maximum; the inputs overlap. Unsigned 32-bit offsets take 64-bit work unless num_items is supplied.
CUB_TEST("Device segmented scan with load balancing handles a total one past the offset maximum",
         "[segmented][scan][device]",
         CUB_LARGE,
         c2h::type_list<cuda::std::int32_t, cuda::std::uint32_t>)
try
{
  using test_offset_t                       = typename c2h::get<0, TestType>;
  constexpr test_offset_t segment_items     = test_offset_t{1} << 30;
  constexpr cuda::std::int64_t total_items  = cuda::std::int64_t{cuda::std::numeric_limits<test_offset_t>::max()} + 1;
  constexpr cuda::std::int64_t num_segments = total_items / segment_items;
  c2h::host_vector<test_offset_t> h_out_begin(static_cast<size_t>(num_segments));
  for (cuda::std::int64_t segment = 0; segment < num_segments; ++segment)
  {
    h_out_begin[segment] = static_cast<test_offset_t>(segment * segment_items);
  }
  const c2h::device_vector<test_offset_t> begins(static_cast<size_t>(num_segments), test_offset_t{0});
  const c2h::device_vector<test_offset_t> ends(static_cast<size_t>(num_segments), segment_items);
  const c2h::device_vector<test_offset_t> out_begin = h_out_begin;
  const test_offset_t* d_begins                     = thrust::raw_pointer_cast(begins.data());
  const test_offset_t* d_ends                       = thrust::raw_pointer_cast(ends.data());
  const test_offset_t* d_out_begin                  = thrust::raw_pointer_cast(out_begin.data());
  const auto d_in                                   = cuda::constant_iterator<item_t>(item_t{1});

  c2h::device_vector<item_t> d_out(static_cast<size_t>(total_items));
  item_t* d_out_ptr = thrust::raw_pointer_cast(d_out.data());

  const auto run = [&](const auto& schedule_env) {
    const size_t bytes = schedule_allocation_size<cub::ForceInclusive::No, false>(
      schedule_env, d_in, d_out_ptr, num_segments, d_begins, d_ends, d_out_begin, cuda::std::plus<>{}, cub::NullType{});
    thrust::fill(c2h::device_policy, d_out.begin(), d_out.end(), item_t{0});
    device_segmented_inclusive_sum(
      d_in,
      d_out_ptr,
      d_begins,
      d_ends,
      d_out_begin,
      num_segments,
      cuda::std::execution::env{expected_allocation_size(bytes), schedule_env});
    const auto expected = cuda::transform_iterator(
      cuda::counting_iterator(cuda::std::int64_t{0}), expected_ones_sum{cuda::std::int64_t{segment_items}});
    REQUIRE(thrust::equal(d_out.cbegin(), d_out.cend(), expected));
  };

  SECTION("with load balancing")
  {
    run(cuda::std::execution::env{cub::segmented_scan_load_balancing});
  }

  SECTION("with load balancing and num_items")
  {
    run(cuda::std::execution::env{cub::segmented_scan_num_items(total_items), cub::segmented_scan_load_balancing});
  }
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
  SUCCEED("exceeding memory is not a failure");
}

// The four output segments are disjoint and cover every int32 position, [-2^31, 2^31), so the total is 2^32, one
// past the uint32_t maximum; the inputs overlap. Load balancing with signed 32-bit offsets requires a total below 2^32
// unless num_items is supplied, in which case a supplied total of 2^32 or more takes 64-bit work.
CUB_TEST("Device segmented scan with load balancing and num_items handles a total of 2^32 with 32-bit offsets",
         "[segmented][scan][device]",
         CUB_LARGE)
try
{
  constexpr offset_t segment_items             = offset_t{1} << 30;
  constexpr cuda::std::int64_t total_items     = cuda::std::int64_t{1} << 32;
  const c2h::device_vector<offset_t> begins    = {0, 0, 0, 0};
  const c2h::device_vector<offset_t> ends      = {segment_items, segment_items, segment_items, segment_items};
  const c2h::device_vector<offset_t> out_begin = {
    cuda::std::numeric_limits<offset_t>::min(), -segment_items, 0, segment_items};
  const offset_t* d_begins                  = thrust::raw_pointer_cast(begins.data());
  const offset_t* d_ends                    = thrust::raw_pointer_cast(ends.data());
  const offset_t* d_out_begin               = thrust::raw_pointer_cast(out_begin.data());
  const auto d_in                           = cuda::constant_iterator<item_t>(item_t{1});
  constexpr cuda::std::int64_t num_segments = 4;

  c2h::device_vector<item_t> d_out(static_cast<size_t>(total_items));
  // Output position 0 is the middle of the buffer, so position -2^31 is its first item.
  item_t* d_out_ptr = thrust::raw_pointer_cast(d_out.data()) + total_items / 2;

  const auto run = [&](const auto& schedule_env) {
    const size_t bytes = schedule_allocation_size<cub::ForceInclusive::No, false>(
      schedule_env, d_in, d_out_ptr, num_segments, d_begins, d_ends, d_out_begin, cuda::std::plus<>{}, cub::NullType{});
    thrust::fill(c2h::device_policy, d_out.begin(), d_out.end(), item_t{0});
    device_segmented_inclusive_sum(
      d_in,
      d_out_ptr,
      d_begins,
      d_ends,
      d_out_begin,
      num_segments,
      cuda::std::execution::env{expected_allocation_size(bytes), schedule_env});
    const auto expected = cuda::transform_iterator(
      cuda::counting_iterator(cuda::std::int64_t{0}), expected_ones_sum{cuda::std::int64_t{segment_items}});
    REQUIRE(thrust::equal(d_out.cbegin(), d_out.cend(), expected));
  };

  SECTION("default")
  {
    run(cuda::std::execution::env{});
  }

  SECTION("with load balancing and num_items")
  {
    run(cuda::std::execution::env{cub::segmented_scan_num_items(total_items), cub::segmented_scan_load_balancing});
  }
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
  SUCCEED("exceeding memory is not a failure");
}

// The four output segments are disjoint and cover 2^32 + 1 positions from -2^31 for signed offsets and from 0 for
// unsigned ones, so the last output position is one past the maximum of a 32-bit offset type; the inputs overlap. The
// output iterator records each write in one bit per position, so no output buffer is allocated, and counts writes
// outside the four segments. The default path scans each segment in its own block and addresses outputs by pointer
// addition; load balancing with a supplied total of 2^32 or more takes 64-bit work and 64-bit output positions.
CUB_TEST("Device segmented scan handles output positions past the 32-bit offset maximum",
         "[segmented][scan][device]",
         CUB_LARGE,
         c2h::type_list<cuda::std::int32_t, cuda::std::uint32_t, cuda::std::int64_t>)
try
{
  using test_offset_t                      = typename c2h::get<0, TestType>;
  constexpr test_offset_t segment_items    = test_offset_t{1} << 30;
  constexpr cuda::std::int64_t total_items = (cuda::std::int64_t{1} << 32) + 1;
  constexpr cuda::std::int64_t first_position =
    cuda::std::is_signed_v<test_offset_t>
      ? cuda::std::int64_t{cuda::std::numeric_limits<cuda::std::int32_t>::min()}
      : cuda::std::int64_t{0};
  const auto output_begin = [](cuda::std::int64_t segment) {
    return static_cast<test_offset_t>(first_position + segment * cuda::std::int64_t{segment_items});
  };
  const c2h::device_vector<test_offset_t> begins    = {0, 0, 0, 0};
  const c2h::device_vector<test_offset_t> ends      = {segment_items, segment_items, segment_items, segment_items + 1};
  const c2h::device_vector<test_offset_t> out_begin = {
    output_begin(0), output_begin(1), output_begin(2), output_begin(3)};
  const test_offset_t* d_begins             = thrust::raw_pointer_cast(begins.data());
  const test_offset_t* d_ends               = thrust::raw_pointer_cast(ends.data());
  const test_offset_t* d_out_begin          = thrust::raw_pointer_cast(out_begin.data());
  const auto d_in                           = cuda::constant_iterator<int>(int{1});
  constexpr cuda::std::int64_t num_segments = 4;

  c2h::device_vector<unsigned long long> stray_writes(1);

  const auto run = [&](const auto& schedule_env) {
    detail::large_problem_test_helper helper(static_cast<std::size_t>(total_items));
    const auto expected = cuda::transform_iterator(
      cuda::counting_iterator(cuda::std::int64_t{0}),
      expected_back_to_back_ones_sum{cuda::std::int64_t{segment_items}, num_segments - 1});
    const auto d_out = cuda::make_tabulate_output_iterator(make_bounded_write_op(
      detail::make_checking_write_op(expected, thrust::raw_pointer_cast(helper.correctness_flags.data())),
      first_position,
      total_items,
      thrust::raw_pointer_cast(stray_writes.data())));
    thrust::fill(c2h::device_policy, stray_writes.begin(), stray_writes.end(), 0ull);
    const size_t bytes = schedule_allocation_size<cub::ForceInclusive::No, false>(
      schedule_env, d_in, d_out, num_segments, d_begins, d_ends, d_out_begin, cuda::std::plus<>{}, cub::NullType{});
    device_segmented_inclusive_sum(
      d_in,
      d_out,
      d_begins,
      d_ends,
      d_out_begin,
      num_segments,
      cuda::std::execution::env{expected_allocation_size(bytes), schedule_env});
    REQUIRE(stray_writes[0] == 0ull);
    helper.check_all_results_correct();
  };

  SECTION("default")
  {
    run(cuda::std::execution::env{});
  }

  SECTION("with load balancing and num_items")
  {
    run(cuda::std::execution::env{cub::segmented_scan_num_items(total_items), cub::segmented_scan_load_balancing});
  }
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
  SUCCEED("exceeding memory is not a failure");
}

// With 32-bit offsets, load balancing keeps 32-bit work below a supplied total of 2^32, and, for signed offsets,
// without num_items. It takes 64-bit work, whose work index needs more temporary storage, from a supplied total of 2^32
// and for unsigned offsets without num_items. With 64-bit offsets the work is 64-bit throughout. Only the temporary
// storage is queried.
CUB_TEST("Device segmented scan with load balancing widens the work only when the total may reach 2^32",
         "[segmented][scan][device]",
         CUB_SMALL,
         c2h::type_list<cuda::std::int32_t, cuda::std::uint32_t, cuda::std::int64_t>)
{
  using test_offset_t                       = typename c2h::get<0, TestType>;
  constexpr cuda::std::int64_t num_segments = 4096;
  constexpr cuda::std::int64_t total_2_32   = cuda::std::int64_t{1} << 32;
  const auto d_begins                       = cuda::constant_iterator<test_offset_t>(test_offset_t{0});
  const auto d_ends                         = cuda::constant_iterator<test_offset_t>(test_offset_t{1});
  const auto d_in                           = cuda::constant_iterator<item_t>(item_t{1});
  item_t* d_out                             = nullptr;

  const auto bytes = [&](const auto& schedule_env) {
    return schedule_allocation_size<cub::ForceInclusive::No, true>(
      schedule_env, d_in, d_out, num_segments, d_begins, d_ends, d_begins, cuda::std::plus<>{}, cub::NullType{});
  };
  const size_t without_total = bytes(cuda::std::execution::env{cub::segmented_scan_load_balancing});
  const size_t below_2_32 =
    bytes(cuda::std::execution::env{cub::segmented_scan_num_items(total_2_32 - 1), cub::segmented_scan_load_balancing});
  const size_t at_2_32 =
    bytes(cuda::std::execution::env{cub::segmented_scan_num_items(total_2_32), cub::segmented_scan_load_balancing});

  if constexpr (sizeof(test_offset_t) == 8)
  {
    REQUIRE(below_2_32 == without_total);
    REQUIRE(at_2_32 == below_2_32);
  }
  else if constexpr (cuda::std::is_signed_v<test_offset_t>)
  {
    REQUIRE(below_2_32 == without_total);
    REQUIRE(at_2_32 > below_2_32);
  }
  else
  {
    REQUIRE(without_total > below_2_32);
    REQUIRE(at_2_32 == without_total);
  }
}

CUB_TEST("Device segmented scan with load balancing works with more segments than the offset type holds",
         "[segmented][scan][device]",
         CUB_LARGE)
try
{
  constexpr cuda::std::int64_t num_segments = cuda::std::int64_t{cuda::std::numeric_limits<offset_t>::max()} + 1000;
  constexpr cuda::std::int64_t num_items    = 3000;
  const auto segment_it                     = cuda::counting_iterator(cuda::std::int64_t{0});
  const auto d_begins                       = cuda::transform_iterator(segment_it, sparse_offset<false>{num_segments});
  const auto d_ends                         = cuda::transform_iterator(segment_it, sparse_offset<true>{num_segments});
  const auto d_in                           = cuda::constant_iterator<item_t>(item_t{1});

  c2h::device_vector<item_t> d_out(num_items);
  item_t* d_out_ptr = thrust::raw_pointer_cast(d_out.data());

  const auto run = [&](const auto& schedule_env) {
    const size_t bytes = schedule_allocation_size<cub::ForceInclusive::No, true>(
      schedule_env, d_in, d_out_ptr, num_segments, d_begins, d_ends, d_begins, cuda::std::plus<>{}, cub::NullType{});
    thrust::fill(c2h::device_policy, d_out.begin(), d_out.end(), item_t{0});
    device_segmented_inclusive_sum(
      d_in,
      d_out_ptr,
      d_begins,
      d_ends,
      num_segments,
      cuda::std::execution::env{expected_allocation_size(bytes), schedule_env});
    const auto expected =
      cuda::transform_iterator(cuda::counting_iterator(cuda::std::int64_t{0}), expected_ones_sum{1000});
    REQUIRE(thrust::equal(d_out.cbegin(), d_out.cend(), expected));
  };

  SECTION("with load balancing")
  {
    run(cuda::std::execution::env{cub::segmented_scan_load_balancing});
  }

  SECTION("with num_items")
  {
    run(cuda::std::execution::env{cub::segmented_scan_num_items(num_items)});
  }
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
  SUCCEED("exceeding memory is not a failure");
}

// num_items = 2^31 over 2^26 segments gives a mean of 32 items, so a block takes at least eight consecutive segments
// for any tile of 128 items or more. The first four segments are nonempty, so they share block 0 and sum to 2^31.
CUB_TEST("Device segmented scan with num_items handles a block whose segments sum past the offset maximum",
         "[segmented][scan][device]",
         CUB_LARGE)
try
{
  constexpr cuda::std::int64_t num_segments = cuda::std::int64_t{1} << 26;
  const auto segment_it                     = cuda::counting_iterator(cuda::std::int64_t{0});
  const auto d_begins                       = cuda::constant_iterator<offset_t>(offset_t{0});
  const auto d_ends                         = cuda::transform_iterator(segment_it, one_block_end_offset{});
  const auto d_out_begin                    = cuda::transform_iterator(segment_it, one_block_output_offset{});
  const auto d_in                           = cuda::constant_iterator<item_t>(item_t{1});

  c2h::device_vector<item_t> d_out(static_cast<size_t>(total_past_max));
  item_t* d_out_ptr = thrust::raw_pointer_cast(d_out.data());

  const auto schedule_env = cuda::std::execution::env{cub::segmented_scan_num_items(total_past_max)};
  const size_t bytes      = schedule_allocation_size<cub::ForceInclusive::No, false>(
    schedule_env, d_in, d_out_ptr, num_segments, d_begins, d_ends, d_out_begin, cuda::std::plus<>{}, cub::NullType{});
  thrust::fill(c2h::device_policy, d_out.begin(), d_out.end(), item_t{0});
  device_segmented_inclusive_sum(
    d_in,
    d_out_ptr,
    d_begins,
    d_ends,
    d_out_begin,
    num_segments,
    cuda::std::execution::env{expected_allocation_size(bytes), schedule_env});

  const auto expected = cuda::transform_iterator(cuda::counting_iterator(cuda::std::int64_t{0}), one_block_expected{});
  REQUIRE(thrust::equal(d_out.cbegin(), d_out.cend(), expected));
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
  SUCCEED("exceeding memory is not a failure");
}

// One segment of 64 items ends at the uint32_t maximum, so any tile of more than 64 items extends past it.
// The output is followed by a guard region that must stay untouched.
CUB_TEST("Device segmented scan handles a segment ending at the offset maximum", "[segmented][scan][device]", CUB_SMALL)
{
  using unsigned_offset_t                               = cuda::std::uint32_t;
  constexpr cuda::std::int64_t segment_items            = 64;
  constexpr cuda::std::int64_t guard_items              = 2048;
  constexpr int guard_value                             = -1;
  constexpr unsigned_offset_t segment_end               = cuda::std::numeric_limits<unsigned_offset_t>::max();
  constexpr unsigned_offset_t segment_begin             = segment_end - static_cast<unsigned_offset_t>(segment_items);
  const c2h::device_vector<unsigned_offset_t> begins    = {segment_begin};
  const c2h::device_vector<unsigned_offset_t> ends      = {segment_end};
  const c2h::device_vector<unsigned_offset_t> out_begin = {0};
  const unsigned_offset_t* d_begins                     = thrust::raw_pointer_cast(begins.data());
  const unsigned_offset_t* d_ends                       = thrust::raw_pointer_cast(ends.data());
  const unsigned_offset_t* d_out_begin                  = thrust::raw_pointer_cast(out_begin.data());
  const auto d_in                                       = cuda::constant_iterator<int>(int{1});
  constexpr cuda::std::int64_t num_segments             = 1;

  c2h::device_vector<int> d_out(static_cast<size_t>(segment_items + guard_items), guard_value);
  int* d_out_ptr = thrust::raw_pointer_cast(d_out.data());

  const auto run = [&](const auto& schedule_env) {
    thrust::fill(c2h::device_policy, d_out.begin(), d_out.end(), guard_value);
    const size_t bytes = schedule_allocation_size<cub::ForceInclusive::No, false>(
      schedule_env, d_in, d_out_ptr, num_segments, d_begins, d_ends, d_out_begin, cuda::std::plus<>{}, cub::NullType{});
    device_segmented_inclusive_sum(
      d_in,
      d_out_ptr,
      d_begins,
      d_ends,
      d_out_begin,
      num_segments,
      cuda::std::execution::env{expected_allocation_size(bytes), schedule_env});

    const auto expected = cuda::counting_iterator(int{1});
    REQUIRE(thrust::equal(d_out.cbegin(), d_out.cbegin() + segment_items, expected));
    const auto guard = cuda::constant_iterator<int>(guard_value);
    REQUIRE(thrust::equal(d_out.cbegin() + segment_items, d_out.cend(), guard));
  };

  run(cuda::std::execution::env{});
  run(cuda::std::execution::env{cub::segmented_scan_num_items(segment_items)});
}

// The three output segments are adjacent in the offset type's wrapped order, so every output segment follows the one
// before it.
CUB_TEST("Device segmented scan keeps output segments either side of the offset wrap apart",
         "[segmented][scan][device]",
         CUB_SMALL,
         c2h::type_list<cuda::std::int32_t, cuda::std::uint32_t>)
{
  using wrap_offset_t                               = typename c2h::get<0, TestType>;
  const c2h::device_vector<wrap_offset_t> begins    = {0, 2, 4};
  const c2h::device_vector<wrap_offset_t> ends      = {2, 4, 5};
  const c2h::device_vector<wrap_offset_t> out_begin = {
    cuda::std::numeric_limits<wrap_offset_t>::max() - 1,
    cuda::std::numeric_limits<wrap_offset_t>::min(),
    cuda::std::numeric_limits<wrap_offset_t>::min() + 2};
  check_outputs_either_side_of_the_wrap(begins, ends, out_begin);
}

// An empty fourth segment whose output begins elsewhere means not every output segment follows the one before it, but
// the first two still share one tile with address deltas that are equal modulo 2^n.
CUB_TEST("Device segmented scan keeps output segments either side of the offset wrap apart within a tile",
         "[segmented][scan][device]",
         CUB_SMALL,
         c2h::type_list<cuda::std::int32_t, cuda::std::uint32_t>)
{
  using wrap_offset_t                               = typename c2h::get<0, TestType>;
  const c2h::device_vector<wrap_offset_t> begins    = {0, 2, 4, 5};
  const c2h::device_vector<wrap_offset_t> ends      = {2, 4, 5, 5};
  const c2h::device_vector<wrap_offset_t> out_begin = {
    cuda::std::numeric_limits<wrap_offset_t>::max() - 1,
    cuda::std::numeric_limits<wrap_offset_t>::min(),
    cuda::std::numeric_limits<wrap_offset_t>::min() + 2,
    cuda::std::numeric_limits<wrap_offset_t>::max() / 2};
  check_outputs_either_side_of_the_wrap(begins, ends, out_begin);
}

// 2^24 items in segments of 1001, 2^24 - 2000 and 999 items, so a block's range spans several tiles and, under CUB's
// default policy, the wrap after the odd first segment falls inside one thread's items in a full tile.
CUB_TEST("Device segmented scan keeps output segments either side of the offset wrap apart across full tiles",
         "[segmented][scan][device]",
         CUB_SMALL,
         c2h::type_list<cuda::std::int32_t, cuda::std::uint32_t>)
{
  using wrap_offset_t = typename c2h::get<0, TestType>;
  const bool gapped   = GENERATE(false, true);
  check_scans_across_the_wrap<wrap_offset_t>(1001, (cuda::std::int64_t{1} << 24) - 2000, 999, gapped, true);
}

// The first output segment holds 2^31 items and ends at the uint32_t maximum and the next two, of 1000 and 1 items,
// follow it from 0, so the total lies between 2^31 and 2^32: below a supplied total of 2^32 the work is 32-bit.
CUB_TEST("Device segmented scan with load balancing handles outputs ending at the unsigned 32-bit offset maximum",
         "[segmented][scan][device]",
         CUB_LARGE)
try
{
  check_scans_across_the_wrap<cuda::std::uint32_t>(cuda::std::int64_t{1} << 31, 1000, 1, false, false);
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
  SUCCEED("exceeding memory is not a failure");
}
