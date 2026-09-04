// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include <cuda/std/__functional/operations.h>
#include <cuda/std/bit>

#include <random>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include <c2h/custom_type.h>
#include <c2h/extended_types.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ReduceByKey, device_reduce_by_key);

// %PARAM% TEST_LAUNCH lid 0:1:2
// %PARAM% TEST_TYPES types 0:1:2:3

// List of types to test
using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t,
                     c2h::equal_comparable_t,
                     c2h::lexicographical_less_comparable_t,
                     c2h::lexicographical_greater_comparable_t>;

#if TEST_TYPES == 0
using full_type_list = c2h::type_list<type_triple<std::uint8_t>, type_triple<std::int8_t, std::int32_t, custom_t>>;
#elif TEST_TYPES == 1
using full_type_list = c2h::type_list<type_triple<std::int32_t>, type_triple<std::int64_t>>;
#elif TEST_TYPES == 2
using full_type_list =
  c2h::type_list<type_triple<uchar3, uchar3, custom_t>,
                 type_triple<
#  if _CCCL_CTK_AT_LEAST(13, 0)
                   ulonglong4_16a
#  else // _CCCL_CTK_AT_LEAST(13, 0)
                   ulonglong4
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
                   >>;
#elif TEST_TYPES == 3
// clang-format off
using full_type_list = c2h::type_list<
type_triple<custom_t>
#if TEST_HALF_T()
, type_triple<half_t> // testing half
#endif // TEST_HALF_T()
#if TEST_BF_T()
, type_triple<bfloat16_t> // testing bf16
#endif // TEST_BF_T()
>;
// clang-format on
#endif

CUB_TEST("Device reduce-by-key works", "[by_key][reduce][device]", CUB_SMALL, full_type_list)
{
  using params   = params_t<TestType>;
  using value_t  = typename params::item_t;
  using output_t = typename params::output_t;
  using key_t    = typename params::type_pair_t::key_t;
  using offset_t = uint32_t;

  constexpr offset_t min_items = 1;
  constexpr offset_t max_items = 1000000;

  // Number of items
  const offset_t num_items = GENERATE_COPY(
    take(2, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));
  INFO("Test num_items: " << num_items);

  // Range of segment sizes to generate (a segment is a series of consecutive equal keys)
  const std::tuple<offset_t, offset_t> seg_size_range =
    GENERATE_COPY(table<offset_t, offset_t>({{1, 1}, {1, num_items}, {num_items, num_items}}));
  INFO("Test seg_size_range: [" << std::get<0>(seg_size_range) << ", " << std::get<1>(seg_size_range) << "]");

  // Generate input segments
  c2h::device_vector<offset_t> segment_offsets = c2h::gen_uniform_offsets<offset_t>(
    C2H_SEED(1), num_items, std::get<0>(seg_size_range), std::get<1>(seg_size_range));

  // Get array of keys from segment offsets
  const offset_t num_segments = static_cast<offset_t>(segment_offsets.size() - 1);
  c2h::device_vector<key_t> segment_keys(num_items);
  c2h::init_key_segments(segment_offsets, segment_keys);
  auto d_keys_it = thrust::raw_pointer_cast(segment_keys.data());

  // Generate input data
  c2h::device_vector<value_t> in_values(num_items);
  c2h::gen(C2H_SEED(2), in_values);
  auto d_values_it = thrust::raw_pointer_cast(in_values.data());

  SECTION("sum")
  {
    using op_t = cuda::std::plus<>;

    // Binary reduction operator
    auto reduction_op = unwrap_op(reference_extended_fp(d_values_it), op_t{});

    // Prepare verification data
    using accum_t = cuda::std::__accumulator_t<op_t, value_t, output_t>;
    c2h::host_vector<output_t> expected_result(num_segments);
    compute_segmented_problem_reference(in_values, segment_offsets, reduction_op, accum_t{}, expected_result.begin());
    c2h::host_vector<key_t> expected_keys = compute_unique_keys_reference(segment_keys);

    // Run test
    c2h::device_vector<offset_t> num_unique_keys(1);
    c2h::device_vector<key_t> out_unique_keys(num_segments);
    c2h::device_vector<output_t> out_result(num_segments);
    auto d_out_it      = thrust::raw_pointer_cast(out_result.data());
    auto d_keys_out_it = thrust::raw_pointer_cast(out_unique_keys.data());
    device_reduce_by_key(
      d_keys_it,
      d_keys_out_it,
      unwrap_it(d_values_it),
      unwrap_it(d_out_it),
      thrust::raw_pointer_cast(num_unique_keys.data()),
      reduction_op,
      num_items);

    // Verify result
    REQUIRE(num_segments == num_unique_keys[0]);
    REQUIRE(expected_result == out_result);
    REQUIRE(expected_keys == out_unique_keys);
  }

  SECTION("min")
  {
    using op_t = cuda::minimum<>;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_segments);
    compute_segmented_problem_reference(
      in_values, segment_offsets, op_t{}, cuda::std::numeric_limits<value_t>::max(), expected_result.begin());
    c2h::host_vector<key_t> expected_keys = compute_unique_keys_reference(segment_keys);

    // Run test
    c2h::device_vector<offset_t> num_unique_keys(1);
    c2h::device_vector<key_t> out_unique_keys(num_segments);
    c2h::device_vector<output_t> out_result(num_segments);
    auto d_result_out_it = thrust::raw_pointer_cast(out_result.data());
    auto d_keys_out_it   = thrust::raw_pointer_cast(out_unique_keys.data());
    device_reduce_by_key(
      d_keys_it,
      d_keys_out_it,
      unwrap_it(d_values_it),
      unwrap_it(d_result_out_it),
      thrust::raw_pointer_cast(num_unique_keys.data()),
      op_t{},
      num_items);

    // Verify result
    REQUIRE(num_segments == num_unique_keys[0]);
    REQUIRE(expected_result == out_result);
    REQUIRE(expected_keys == out_unique_keys);
  }
}

#if TEST_LAUNCH == 0 && TEST_TYPES == 0
CUB_TEST_CASE("Device reduce-by-key is run-to-run deterministic for fp64 sums", "[by_key][reduce][device]", CUB_SMALL)
{
  constexpr std::size_t num_items = 65'536;
  constexpr std::size_t long_run  = 50'000;
  constexpr std::size_t short_run = 64;
  constexpr int num_repetitions   = 20;

  c2h::host_vector<std::uint64_t> keys(num_items);
  c2h::host_vector<double> values(num_items);
  std::mt19937_64 rng{42};
  std::uniform_real_distribution<double> distribution{-1.0, 1.0};

  for (std::size_t i = 0; i < num_items; ++i)
  {
    keys[i]   = i < long_run ? 0 : 1 + (i - long_run) / short_run;
    values[i] = i % 7 == 0 ? distribution(rng) * 1e-2 : 0.0;
    static_cast<void>(rng()); // Match the issue's integer-control RNG consumption.
  }

  c2h::device_vector<std::uint64_t> keys_in = keys;
  c2h::device_vector<double> values_in      = values;
  c2h::device_vector<std::uint64_t> unique_out(num_items);
  c2h::device_vector<double> aggregates_out(num_items);
  c2h::device_vector<std::size_t> num_runs_out(1);

  auto* const d_keys_in        = thrust::raw_pointer_cast(keys_in.data());
  auto* const d_values_in      = thrust::raw_pointer_cast(values_in.data());
  auto* const d_unique_out     = thrust::raw_pointer_cast(unique_out.data());
  auto* const d_aggregates_out = thrust::raw_pointer_cast(aggregates_out.data());
  auto* const d_num_runs_out   = thrust::raw_pointer_cast(num_runs_out.data());

  std::size_t temp_storage_bytes{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::ReduceByKey(
      nullptr,
      temp_storage_bytes,
      d_keys_in,
      d_unique_out,
      d_values_in,
      d_aggregates_out,
      d_num_runs_out,
      cuda::std::plus{},
      num_items));
  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  auto* const d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  std::uint64_t first_result{};
  for (int repetition = 0; repetition < num_repetitions; ++repetition)
  {
    REQUIRE(cudaSuccess == cudaMemset(d_aggregates_out, 0xee, sizeof(double)));
    REQUIRE(
      cudaSuccess
      == cub::DeviceReduce::ReduceByKey(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in,
        d_unique_out,
        d_values_in,
        d_aggregates_out,
        d_num_runs_out,
        cuda::std::plus{},
        num_items));

    const auto result = cuda::std::bit_cast<std::uint64_t>(static_cast<double>(aggregates_out[0]));
    if (repetition == 0)
    {
      first_result = result;
    }
    else
    {
      REQUIRE(result == first_result);
    }
  }
}
#endif // TEST_LAUNCH == 0 && TEST_TYPES == 0
