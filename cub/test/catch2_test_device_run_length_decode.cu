// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_run_length_decode.cuh>

#include <cuda/iterator>
#include <cuda/std/cstdint>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <new>
#include <numeric>

#include "catch2_large_problem_helper.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceRunLengthDecode::Decode, run_length_decode);
DECLARE_LAUNCH_WRAPPER(cub::DeviceRunLengthDecode::DecodeFromOffsets, run_length_decode_from_offsets);

using value_types = c2h::type_list<std::uint8_t, std::uint64_t, c2h::custom_type_t<c2h::equal_comparable_t>>;

// std::uint8_t and std::uint16_t are narrower than the length type the runs are copied with
using run_length_types = c2h::type_list<std::uint8_t, std::uint16_t, std::int32_t, std::int64_t>;
using run_offset_types = c2h::type_list<std::uint32_t, std::int64_t>;

template <typename T, typename RunLengthT>
c2h::host_vector<T>
reference_decode(const c2h::host_vector<T>& run_values, const c2h::host_vector<RunLengthT>& run_lengths)
{
  const auto num_items = std::accumulate(run_lengths.begin(), run_lengths.end(), std::size_t{0});
  c2h::host_vector<T> decoded(num_items, thrust::default_init);
  auto decoded_it = decoded.begin();
  for (std::size_t run = 0; run < run_values.size(); ++run)
  {
    decoded_it = std::fill_n(decoded_it, run_lengths[run], run_values[run]);
  }
  return decoded;
}

template <typename RunOffsetT, typename RunLengthT>
c2h::host_vector<RunOffsetT> reference_run_offsets(const c2h::host_vector<RunLengthT>& run_lengths)
{
  c2h::host_vector<RunOffsetT> run_offsets(run_lengths.size() + 1, thrust::no_init);
  run_offsets[0] = 0;
  std::partial_sum(run_lengths.begin(), run_lengths.end(), run_offsets.begin() + 1);
  return run_offsets;
}

// Runs of up to 128 items are copied by single threads, runs of up to 8K items by warps, and longer runs by one or
// more thread blocks (with the default tuning of the batched copy). The run lengths are drawn from [0, max_run_length],
// and at most max_num_items / max_run_length runs are drawn, which bounds the decoded size by max_num_items.
inline constexpr int max_num_items = 1 << 22;

template <typename T, typename RunLengthT>
void test_decode(c2h::seed_t values_seed, c2h::seed_t lengths_seed, int num_runs, RunLengthT max_run_length)
{
  c2h::device_vector<T> d_run_values(num_runs, thrust::default_init);
  c2h::device_vector<RunLengthT> d_run_lengths(num_runs, thrust::no_init);
  c2h::gen(values_seed, d_run_values);
  c2h::gen(lengths_seed, d_run_lengths, RunLengthT{0}, max_run_length);

  const c2h::host_vector<T> expected =
    reference_decode(c2h::host_vector<T>(d_run_values), c2h::host_vector<RunLengthT>(d_run_lengths));
  c2h::device_vector<T> d_out(expected.size(), thrust::default_init);

  run_length_decode(d_run_values.begin(), d_run_lengths.begin(), d_out.begin(), num_runs);

  REQUIRE(expected == d_out);
}

template <typename T, typename RunOffsetT>
void test_decode_from_offsets(c2h::seed_t values_seed, c2h::seed_t lengths_seed, int num_runs, RunOffsetT max_run_length)
{
  c2h::device_vector<T> d_run_values(num_runs, thrust::default_init);
  c2h::device_vector<RunOffsetT> d_run_lengths(num_runs, thrust::no_init);
  c2h::gen(values_seed, d_run_values);
  c2h::gen(lengths_seed, d_run_lengths, RunOffsetT{0}, max_run_length);

  const c2h::host_vector<RunOffsetT> h_run_lengths(d_run_lengths);
  const c2h::device_vector<RunOffsetT> d_run_offsets = reference_run_offsets<RunOffsetT>(h_run_lengths);
  const c2h::host_vector<T> expected = reference_decode(c2h::host_vector<T>(d_run_values), h_run_lengths);
  c2h::device_vector<T> d_out(expected.size(), thrust::default_init);

  run_length_decode_from_offsets(d_run_values.begin(), d_run_offsets.begin(), d_out.begin(), num_runs);

  REQUIRE(expected == d_out);
}

CUB_TEST("DeviceRunLengthDecode::Decode works", "[device][run_length_decode]", CUB_SMALL, value_types)
{
  using value_t      = c2h::get<0, TestType>;
  using run_length_t = std::int32_t;

  const run_length_t max_run_length = GENERATE(run_length_t{4}, run_length_t{1000}, run_length_t{20000});
  const int num_runs = GENERATE_COPY(take(2, random(1, max_num_items / static_cast<int>(max_run_length))));

  const c2h::seed_t values_seed  = C2H_SEED(1);
  const c2h::seed_t lengths_seed = C2H_SEED(1);

  test_decode<value_t>(values_seed, lengths_seed, num_runs, max_run_length);
}

CUB_TEST("DeviceRunLengthDecode::Decode works with different run length types",
         "[device][run_length_decode]",
         CUB_SMALL,
         run_length_types)
{
  using value_t      = std::int32_t;
  using run_length_t = c2h::get<0, TestType>;

  // Thread-, warp- and block-level copies, as far as the run length type can represent the maximum run length
  const auto max_run_length = static_cast<run_length_t>(GENERATE(filter(
    [](std::uint64_t max_length) {
      return max_length <= std::uint64_t{(std::numeric_limits<run_length_t>::max)()};
    },
    values({std::uint64_t{4}, std::uint64_t{255}, std::uint64_t{20000}}))));
  const int num_runs        = GENERATE_COPY(take(2, random(1, max_num_items / static_cast<int>(max_run_length))));

  const c2h::seed_t values_seed  = C2H_SEED(1);
  const c2h::seed_t lengths_seed = C2H_SEED(1);

  test_decode<value_t>(values_seed, lengths_seed, num_runs, max_run_length);
}

CUB_TEST("DeviceRunLengthDecode::DecodeFromOffsets works", "[device][run_length_decode]", CUB_SMALL, value_types)
{
  using value_t      = c2h::get<0, TestType>;
  using run_offset_t = std::int32_t;

  const run_offset_t max_run_length = GENERATE(run_offset_t{4}, run_offset_t{1000}, run_offset_t{20000});
  const int num_runs = GENERATE_COPY(take(2, random(1, max_num_items / static_cast<int>(max_run_length))));

  const c2h::seed_t values_seed  = C2H_SEED(1);
  const c2h::seed_t lengths_seed = C2H_SEED(1);

  test_decode_from_offsets<value_t>(values_seed, lengths_seed, num_runs, max_run_length);
}

CUB_TEST("DeviceRunLengthDecode::DecodeFromOffsets works with different offset types",
         "[device][run_length_decode]",
         CUB_SMALL,
         run_offset_types)
{
  using value_t      = std::int32_t;
  using run_offset_t = c2h::get<0, TestType>;

  const run_offset_t max_run_length = GENERATE(run_offset_t{4}, run_offset_t{1000}, run_offset_t{20000});
  const int num_runs = GENERATE_COPY(take(2, random(1, max_num_items / static_cast<int>(max_run_length))));

  const c2h::seed_t values_seed  = C2H_SEED(1);
  const c2h::seed_t lengths_seed = C2H_SEED(1);

  test_decode_from_offsets<value_t>(values_seed, lengths_seed, num_runs, max_run_length);
}

// Zero-length runs at the start, consecutively in the middle, and at the end, which the random tests only hit by chance
CUB_TEST("DeviceRunLengthDecode::Decode handles runs of length zero", "[device][run_length_decode]", CUB_SMALL)
{
  const c2h::device_vector<int> d_run_values{1, 2, 3, 4, 5, 6, 7};
  const c2h::device_vector<int> d_run_lengths{0, 3, 0, 0, 2, 1, 0};
  const int num_runs = static_cast<int>(d_run_values.size());

  c2h::device_vector<int> d_out(6, thrust::no_init);
  run_length_decode(d_run_values.begin(), d_run_lengths.begin(), d_out.begin(), num_runs);

  const c2h::device_vector<int> expected{2, 2, 2, 5, 5, 6};
  REQUIRE(expected == d_out);
}

CUB_TEST("DeviceRunLengthDecode::DecodeFromOffsets handles runs of length zero",
         "[device][run_length_decode]",
         CUB_SMALL)
{
  const c2h::device_vector<int> d_run_values{1, 2, 3, 4, 5, 6, 7};
  const c2h::device_vector<int> d_run_offsets{0, 0, 3, 3, 3, 5, 6, 6};
  const int num_runs = static_cast<int>(d_run_values.size());

  c2h::device_vector<int> d_out(6, thrust::no_init);
  run_length_decode_from_offsets(d_run_values.begin(), d_run_offsets.begin(), d_out.begin(), num_runs);

  const c2h::device_vector<int> expected{2, 2, 2, 5, 5, 6};
  REQUIRE(expected == d_out);
}

CUB_TEST("DeviceRunLengthDecode::Decode handles only runs of length zero", "[device][run_length_decode]", CUB_SMALL)
{
  const c2h::device_vector<int> d_run_values{1, 2, 3};
  const c2h::device_vector<int> d_run_lengths{0, 0, 0};
  const int num_runs = static_cast<int>(d_run_values.size());

  // Nothing may be written
  c2h::device_vector<int> d_out(1, 42);
  run_length_decode(d_run_values.begin(), d_run_lengths.begin(), d_out.begin(), num_runs);

  REQUIRE(d_out[0] == 42);
}

CUB_TEST("DeviceRunLengthDecode::Decode handles zero runs", "[device][run_length_decode]", CUB_SMALL)
{
  const c2h::device_vector<int> d_run_values{};
  const c2h::device_vector<int> d_run_lengths{};

  // Nothing may be written
  c2h::device_vector<int> d_out(1, 42);
  run_length_decode(d_run_values.begin(), d_run_lengths.begin(), d_out.begin(), 0);

  REQUIRE(d_out[0] == 42);
}

CUB_TEST("DeviceRunLengthDecode::DecodeFromOffsets handles zero runs", "[device][run_length_decode]", CUB_SMALL)
{
  const c2h::device_vector<int> d_run_values{};
  const c2h::device_vector<int> d_run_offsets{0};

  // Nothing may be written
  c2h::device_vector<int> d_out(1, 42);
  run_length_decode_from_offsets(d_run_values.begin(), d_run_offsets.begin(), d_out.begin(), 0);

  REQUIRE(d_out[0] == 42);
}

CUB_TEST("DeviceRunLengthDecode::DecodeFromOffsets writes only between the first and the last offset",
         "[device][run_length_decode]",
         CUB_SMALL)
{
  const c2h::device_vector<int> d_run_values{1, 2, 3};
  const c2h::device_vector<int> d_run_offsets{2, 4, 5, 8};
  const int num_runs = static_cast<int>(d_run_values.size());

  c2h::device_vector<int> d_out(10, 42);
  run_length_decode_from_offsets(d_run_values.begin(), d_run_offsets.begin(), d_out.begin(), num_runs);

  const c2h::device_vector<int> expected{42, 42, 1, 1, 2, 3, 3, 3, 42, 42};
  REQUIRE(expected == d_out);
}

// Truncates the index of a run to its value
struct run_index_to_value_op
{
  __host__ __device__ std::uint8_t operator()(std::int64_t run) const
  {
    return static_cast<std::uint8_t>(run);
  }
};

// Expected value of a decoded item for runs of equal length whose values are given by run_index_to_value_op
struct equal_length_runs_expected_value_op
{
  std::int64_t run_length;

  __host__ __device__ std::uint8_t operator()(std::int64_t item) const
  {
    return run_index_to_value_op{}(item / run_length);
  }
};

// Expected value of a decoded item for three runs with the values 1, 2, and 3
struct three_runs_expected_value_op
{
  std::int64_t second_run_begin;
  std::int64_t third_run_begin;

  __host__ __device__ std::uint8_t operator()(std::int64_t item) const
  {
    return item < second_run_begin ? std::uint8_t{1} : item < third_run_begin ? std::uint8_t{2} : std::uint8_t{3};
  }
};

CUB_TEST("DeviceRunLengthDecode::Decode works for more than 2^32 decoded items",
         "[device][run_length_decode][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
         CUB_LARGE)
try
{
  // Runs of 2^20 items with 32-bit run lengths, so the offsets of the runs following the first 2^12 runs exceed 32 bits
  using run_length_t                = std::uint32_t;
  constexpr run_length_t run_length = run_length_t{1} << 20;
  constexpr std::int64_t num_runs   = (std::int64_t{1} << 12) + 3;
  constexpr std::int64_t num_items  = num_runs * run_length;

  const auto d_run_values = cuda::transform_iterator(cuda::counting_iterator(std::int64_t{0}), run_index_to_value_op{});
  const auto d_run_lengths = cuda::constant_iterator(run_length);

  auto check_result_helper = detail::large_problem_test_helper(static_cast<std::size_t>(num_items));
  const auto d_out         = check_result_helper.get_flagging_output_iterator(cuda::transform_iterator(
    cuda::counting_iterator(std::int64_t{0}), equal_length_runs_expected_value_op{run_length}));

  run_length_decode(d_run_values, d_run_lengths, d_out, num_runs);

  check_result_helper.check_all_results_correct();
}
catch (std::bad_alloc& e)
{
  std::cerr << "Caught bad_alloc: " << e.what() << '\n';
}

CUB_TEST("DeviceRunLengthDecode::Decode works for a run of more than 2^32 items",
         "[device][run_length_decode][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
         CUB_LARGE)
try
{
  constexpr std::int64_t second_run_length = (std::int64_t{1} << 32) + 7;
  constexpr std::int64_t num_items         = 5 + second_run_length + 3;

  const c2h::device_vector<std::uint8_t> d_run_values{1, 2, 3};
  const c2h::device_vector<std::int64_t> d_run_lengths{5, second_run_length, 3};
  const std::int64_t num_runs = 3;

  auto check_result_helper = detail::large_problem_test_helper(static_cast<std::size_t>(num_items));
  const auto d_out         = check_result_helper.get_flagging_output_iterator(cuda::transform_iterator(
    cuda::counting_iterator(std::int64_t{0}), three_runs_expected_value_op{5, 5 + second_run_length}));

  run_length_decode(d_run_values.begin(), d_run_lengths.begin(), d_out, num_runs);

  check_result_helper.check_all_results_correct();
}
catch (std::bad_alloc& e)
{
  std::cerr << "Caught bad_alloc: " << e.what() << '\n';
}

CUB_TEST("DeviceRunLengthDecode::DecodeFromOffsets works for more than 2^32 decoded items",
         "[device][run_length_decode][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck]",
         CUB_LARGE)
try
{
  // The second run has more than 2^32 items and the third run starts beyond 2^32 items
  constexpr std::int64_t second_run_length = (std::int64_t{1} << 32) + 7;
  constexpr std::int64_t num_items         = 5 + second_run_length + 3;

  const c2h::device_vector<std::uint8_t> d_run_values{1, 2, 3};
  const c2h::device_vector<std::int64_t> d_run_offsets{0, 5, 5 + second_run_length, num_items};
  const std::int64_t num_runs = 3;

  auto check_result_helper = detail::large_problem_test_helper(static_cast<std::size_t>(num_items));
  const auto d_out         = check_result_helper.get_flagging_output_iterator(cuda::transform_iterator(
    cuda::counting_iterator(std::int64_t{0}), three_runs_expected_value_op{5, 5 + second_run_length}));

  run_length_decode_from_offsets(d_run_values.begin(), d_run_offsets.begin(), d_out, num_runs);

  check_result_helper.check_all_results_correct();
}
catch (std::bad_alloc& e)
{
  std::cerr << "Caught bad_alloc: " << e.what() << '\n';
}
