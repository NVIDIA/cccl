// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_run_length_encode.cuh>

#include <thrust/logical.h>
#include <thrust/sequence.h>

#include <cuda/iterator>

#include <algorithm>
#include <limits>
#include <numeric>

#include "catch2_large_problem_helper.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceRunLengthEncode::NonTrivialRuns, run_length_encode);

// %PARAM% TEST_LAUNCH lid 0:1:2

using all_types =
  c2h::type_list<std::uint8_t,
                 std::uint64_t,
                 std::int8_t,
                 std::int64_t,
                 ulonglong2,
                 c2h::custom_type_t<c2h::equal_comparable_t>>;

using types = c2h::type_list<std::uint32_t, std::int8_t>;

// List of offset types to be used for testing large number of items
using offset_types = c2h::type_list<std::int64_t, std::uint32_t, std::int32_t>;

// generates for [0, 1, 2, ...] the sequence [0, 1, 1, 2, 3, 3, 4, 5, 5, ...]
struct index_to_item_op
{
  template <typename OffsetT>
  __host__ __device__ OffsetT operator()(const OffsetT index) const
  {
    // Calculate the number at the given index
    // The pattern repeats: odd, odd, even
    // For every 3 indices: [odd, odd, even]
    OffsetT group = index / OffsetT{3};
    OffsetT pos   = index % OffsetT{3};
    if (pos == OffsetT{0})
    {
      // Even number (once)
      return (OffsetT{2} * group);
    }
    else
    {
      // Odd number (repeated twice)
      return OffsetT{2} * group + OffsetT{1};
    }
  }
};

struct run_index_to_offset_op
{
  template <typename OffsetT>
  __host__ __device__ OffsetT operator()(OffsetT run_index)
  {
    return OffsetT{1} + OffsetT{3} * run_index;
  }
};

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle empty input", "[device][run_length_encode]")
{
  constexpr int num_items = 0;
  c2h::device_vector<int> out_num_runs(1, 42);

  // Note intentionally no discard_iterator as we want to ensure nothing is written to the output arrays
  run_length_encode(
    static_cast<int*>(nullptr),
    static_cast<int*>(nullptr),
    static_cast<int*>(nullptr),
    thrust::raw_pointer_cast(out_num_runs.data()),
    num_items);

  REQUIRE(out_num_runs.front() == 0);
}

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle a single element", "[device][run_length_encode]")
{
  constexpr int num_items = 1;
  c2h::device_vector<int> out_num_runs(1, 42);

  // Note intentionally no discard_iterator as we want to ensure nothing is written to the output arrays
  run_length_encode(
    static_cast<int*>(nullptr),
    static_cast<int*>(nullptr),
    static_cast<int*>(nullptr),
    thrust::raw_pointer_cast(out_num_runs.data()),
    num_items);

  REQUIRE(out_num_runs.front() == 0);
}

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle different counting types", "[device][run_length_encode]")
{
  constexpr int num_items = 1;
  c2h::device_vector<int> in(num_items, 42);
  c2h::device_vector<int> out_num_runs(1, 42);

  // Note intentionally no discard_iterator as we want to ensure nothing is written to the output
  // arrays
  run_length_encode(
    in.begin(),
    static_cast<cuda::std::size_t*>(nullptr),
    static_cast<std::uint16_t*>(nullptr),
    out_num_runs.begin(),
    num_items);

  REQUIRE(out_num_runs.front() == 0);
}

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle all unique", "[device][run_length_encode]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 10;
  c2h::device_vector<int> out_num_runs(1, -1);

  run_length_encode(
    cuda::make_counting_iterator(type{}),
    static_cast<int*>(nullptr),
    static_cast<int*>(nullptr),
    out_num_runs.begin(),
    num_items);

  REQUIRE(out_num_runs.front() == 0);
}

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle all equal", "[device][run_length_encode]", types)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int num_items = 10;
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<int> out_offsets(1, -1);
  c2h::device_vector<int> out_lengths(1, -1);
  c2h::device_vector<int> out_num_runs(1, -1);
  c2h::gen(C2H_SEED(2), in);
  thrust::fill(c2h::device_policy, in.begin(), in.end(), in.front());

  run_length_encode(in.begin(), out_offsets.begin(), out_lengths.begin(), out_num_runs.begin(), num_items);

  REQUIRE(out_offsets.front() == 0);
  REQUIRE(out_lengths.front() == num_items);
  REQUIRE(out_num_runs.front() == 1);
}

template <class T, class Index>
bool validate_results(
  const c2h::device_vector<T>& in,
  const c2h::device_vector<Index>& out_offsets,
  const c2h::device_vector<Index>& out_lengths,
  const c2h::device_vector<Index>& out_num_runs,
  const int num_items)
{
  const c2h::host_vector<T>& h_in               = in;
  const c2h::host_vector<Index>& h_out_offsets  = out_offsets;
  const c2h::host_vector<Index>& h_out_lengths  = out_lengths;
  const c2h::host_vector<Index>& h_out_num_runs = out_num_runs;

  const cuda::std::size_t num_runs = static_cast<cuda::std::size_t>(h_out_num_runs.front());
  for (cuda::std::size_t run = 0; run < num_runs; ++run)
  {
    const cuda::std::size_t first_index = static_cast<cuda::std::size_t>(h_out_offsets[run]);
    const cuda::std::size_t final_index = first_index + static_cast<cuda::std::size_t>(h_out_lengths[run]);

    // Ensure we started a new run
    if (first_index > 0)
    {
      if (h_in[first_index] == h_in[first_index - 1])
      {
        return false;
      }
    }

    // Ensure the run is valid
    const auto first_elem = h_in[first_index];
    const auto all_equal  = [first_elem](const T& elem) -> bool {
      return first_elem == elem;
    };
    if (!std::all_of(h_in.begin() + first_index + 1, h_in.begin() + final_index, all_equal))
    {
      return false;
    }

    // Ensure the run is of maximal length
    if (final_index < static_cast<cuda::std::size_t>(num_items))
    {
      if (h_in[first_index] == h_in[final_index])
      {
        return false;
      }
    }
  }
  return true;
}

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle iterators", "[device][run_length_encode]", all_types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<int> out_offsets(num_items, -1);
  c2h::device_vector<int> out_lengths(num_items, -1);
  c2h::device_vector<int> out_num_runs(1, -1);
  c2h::gen(C2H_SEED(2), in);

  run_length_encode(in.begin(), out_offsets.begin(), out_lengths.begin(), out_num_runs.begin(), num_items);

  out_offsets.resize(out_num_runs.front());
  out_lengths.resize(out_num_runs.front());
  REQUIRE(validate_results(in, out_offsets, out_lengths, out_num_runs, num_items));
}

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns can handle pointers", "[device][run_length_encode]", types)
{
  using type = typename c2h::get<0, TestType>;

  const int num_items = GENERATE(take(2, random(1, 1000000)));
  c2h::device_vector<type> in(num_items);
  c2h::device_vector<int> out_offsets(num_items, -1);
  c2h::device_vector<int> out_lengths(num_items, -1);
  c2h::device_vector<int> out_num_runs(1, -1);
  c2h::gen(C2H_SEED(2), in);

  run_length_encode(
    thrust::raw_pointer_cast(in.data()),
    thrust::raw_pointer_cast(out_offsets.data()),
    thrust::raw_pointer_cast(out_lengths.data()),
    thrust::raw_pointer_cast(out_num_runs.data()),
    num_items);

  out_offsets.resize(out_num_runs.front());
  out_lengths.resize(out_num_runs.front());
  REQUIRE(validate_results(in, out_offsets, out_lengths, out_num_runs, num_items));
}

// Guard against #293
template <bool TimeSlicing>
struct device_rle_policy_hub
{
  static constexpr int threads = 96;
  static constexpr int items   = 15;

  struct Policy500 : cub::ChainedPolicy<500, Policy500, Policy500>
  {
    using RleSweepPolicyT = cub::
      AgentRlePolicy<threads, items, cub::BLOCK_LOAD_DIRECT, cub::LOAD_DEFAULT, TimeSlicing, cub::BLOCK_SCAN_WARP_SCANS>;
  };

  using MaxPolicy = Policy500;
};

struct CustomDeviceRunLengthEncode
{
  template <bool TimeSlicing,
            typename InputIteratorT,
            typename OffsetsOutputIteratorT,
            typename LengthsOutputIteratorT,
            typename NumRunsOutputIteratorT>
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t NonTrivialRuns(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OffsetsOutputIteratorT d_offsets_out,
    LengthsOutputIteratorT d_lengths_out,
    NumRunsOutputIteratorT d_num_runs_out,
    int num_items,
    cudaStream_t stream = 0)
  {
    using OffsetT    = int; // Signed integer type for global offsets
    using EqualityOp = cuda::std::equal_to<>; // Default == operator

    return cub::DeviceRleDispatch<InputIteratorT,
                                  OffsetsOutputIteratorT,
                                  LengthsOutputIteratorT,
                                  NumRunsOutputIteratorT,
                                  EqualityOp,
                                  OffsetT,
                                  device_rle_policy_hub<TimeSlicing>>::
      Dispatch(d_temp_storage,
               temp_storage_bytes,
               d_in,
               d_offsets_out,
               d_lengths_out,
               d_num_runs_out,
               EqualityOp(),
               num_items,
               stream);
  }
};

DECLARE_LAUNCH_WRAPPER(CustomDeviceRunLengthEncode::NonTrivialRuns<true>, run_length_encode_293_true);
DECLARE_LAUNCH_WRAPPER(CustomDeviceRunLengthEncode::NonTrivialRuns<false>, run_length_encode_293_false);

using time_slicing = c2h::type_list<std::true_type, std::false_type>;

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns does not run out of memory", "[device][run_length_encode]", time_slicing)
{
  using type         = typename c2h::get<0, TestType>;
  using policy_hub_t = device_rle_policy_hub<type::value>;

  constexpr int tile_size    = policy_hub_t::threads * policy_hub_t::items;
  constexpr int num_items    = 2 * tile_size;
  constexpr int magic_number = num_items + 1;

  c2h::host_vector<int> h_keys(num_items);
  thrust::sequence(h_keys.begin(), h_keys.begin() + tile_size);

  int expected_non_trivial_runs = 0;
  int value                     = tile_size;
  int large_group_size          = 3;
  for (int i = 0; i < tile_size; i++)
  {
    int j = 0;
    for (; j < large_group_size && i < tile_size; ++j, ++i)
    {
      h_keys[tile_size + i] = value;
    }
    if (j == large_group_size)
    {
      ++expected_non_trivial_runs;
    }
    ++value;

    if (i < tile_size)
    {
      h_keys[tile_size + i] = value;
    }
    ++value;
  }

  // in #293 we were writing before the output arrays. So add a sentinel element in front to check
  // against OOB writes
  c2h::device_vector<int> in = h_keys;
  c2h::device_vector<int> out_offsets(num_items + 1, -1);
  c2h::device_vector<int> out_lengths(num_items + 1, -1);
  c2h::device_vector<int> out_num_runs(1, -1);
  out_offsets.front() = magic_number;
  out_lengths.front() = magic_number;

  if constexpr (type::value)
  {
    run_length_encode_293_true(
      in.begin(), out_offsets.begin() + 1, out_lengths.begin() + 1, out_num_runs.begin(), num_items);
  }
  else
  {
    run_length_encode_293_false(
      in.begin(), out_offsets.begin() + 1, out_lengths.begin() + 1, out_num_runs.begin(), num_items);
  }

  REQUIRE(out_num_runs.front() == expected_non_trivial_runs);
  REQUIRE(out_lengths.front() == magic_number);
  REQUIRE(out_offsets.front() == magic_number);
}

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns works for a large number of items",
         "[device][run_length_encode][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck][!mayfail]",
         offset_types)
try
{
  using offset_type     = typename c2h::get<0, TestType>;
  using run_length_type = offset_type;

  cuda::std::size_t extra_items = GENERATE(take(1, random((1 << 20), (1 << 22))));
  const auto num_items          = detail::make_large_offset<offset_type>(extra_items);
  CAPTURE(c2h::type_name<offset_type>(), c2h::type_name<run_length_type>(), num_items);

  auto counting_it = cuda::make_counting_iterator(offset_type{0});

  // Input iterator: repeat odd numbers once and even numbers twice
  auto input_item_it = cuda::make_transform_iterator(counting_it, index_to_item_op{});
  // Number of non-trivial runs is number of full(!) three-item (i.e., even, odd, odd) groups
  const auto num_uniques = num_items / 3;

  // Prepare helper to check the unique items being written: we expect the i-th item corresponding to value i
  auto check_offset_out_helper = detail::large_problem_test_helper(num_uniques);
  auto expected_offsets_it     = cuda::make_transform_iterator(counting_it, run_index_to_offset_op{});
  auto check_offset_out_it     = check_offset_out_helper.get_flagging_output_iterator(expected_offsets_it);

  // Prepare helper to check the run-lengths being written: i-th item corresponding to value i
  auto check_run_length_out_helper = detail::large_problem_test_helper(num_uniques);
  auto expected_run_lengths_it     = cuda::make_constant_iterator(run_length_type{2});
  auto check_run_length_out_it     = check_run_length_out_helper.get_flagging_output_iterator(expected_run_lengths_it);

  // Allocate memory for the number of expected unique items
  c2h::device_vector<offset_type> out_num_runs(1);

  // Run algorithm under test
  run_length_encode(input_item_it, check_offset_out_it, check_run_length_out_it, out_num_runs.begin(), num_items);

  // Verify result
  CHECK(out_num_runs[0] == num_uniques);
  check_offset_out_helper.check_all_results_correct();
  check_run_length_out_helper.check_all_results_correct();
}
catch (const std::bad_alloc& e)
{
  std::cerr << "Caught bad_alloc: " << e.what() << std::endl;
}

C2H_TEST("DeviceRunLengthEncode::NonTrivialRuns works for large runs of equal items",
         "[device][run_length_encode][skip-cs-initcheck][skip-cs-racecheck][skip-cs-synccheck][!mayfail]",
         offset_types)
try
{
  using offset_type     = typename c2h::get<0, TestType>;
  using run_length_type = offset_type;
  using item_t          = char;

  CAPTURE(c2h::type_name<offset_type>(), c2h::type_name<run_length_type>(), c2h::type_name<item_t>());

  const auto num_items = detail::make_large_offset<offset_type>();

  constexpr auto num_uniques               = offset_type{2};
  constexpr run_length_type first_run_size = 200;
  const run_length_type second_run_size    = static_cast<run_length_type>(num_items) - first_run_size;

  // First run is a small run of equal items
  auto small_segment_it = cuda::make_constant_iterator(item_t{3});
  // Second run is a very large run of equal items
  auto large_segment_it = cuda::make_constant_iterator(item_t{42});
  auto input_item_it    = detail::make_concat_iterators_op(small_segment_it, large_segment_it, first_run_size);

  // Allocate some memory for the results
  c2h::device_vector<offset_type> offsets_out(num_uniques, thrust::no_init);
  c2h::device_vector<run_length_type> run_lengths_out(num_uniques, thrust::no_init);

  // Allocate memory for the number of expected unique items
  c2h::device_vector<offset_type> out_num_runs(1);

  // Run algorithm under test
  run_length_encode(
    input_item_it,
    offsets_out.begin(),
    run_lengths_out.begin(),
    out_num_runs.begin(),
    static_cast<offset_type>(num_items));

  // Expected results
  c2h::device_vector<offset_type> expected_uniques{offset_type{0}, first_run_size};
  c2h::device_vector<run_length_type> expected_run_lengths{first_run_size, second_run_size};

  // Verify result
  CHECK(out_num_runs[0] == num_uniques);
  CHECK(expected_uniques == offsets_out);
  CHECK(expected_run_lengths == run_lengths_out);
}
catch (const std::bad_alloc& e)
{
  std::cerr << "Caught bad_alloc: " << e.what() << std::endl;
}
