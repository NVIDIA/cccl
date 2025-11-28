// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_device_scan.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScanInit, device_inclusive_scan_with_init);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveSum, device_exclusive_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveScan, device_exclusive_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveSum, device_inclusive_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScan, device_inclusive_scan);

// %PARAM% TEST_LAUNCH lid 0:1:2
// %PARAM% TEST_TYPES types 0:1:2:3

// List of types to test
using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t,
                     c2h::equal_comparable_t,
                     c2h::lexicographical_less_comparable_t,
                     c2h::lexicographical_greater_comparable_t>;

#if TEST_TYPES == 0
using full_type_list = c2h::type_list<type_pair<int>>;
#elif TEST_TYPES == 1
using full_type_list = c2h::type_list<type_pair<std::int32_t>, type_pair<std::uint64_t>>;
#elif TEST_TYPES == 2
using full_type_list =
  c2h::type_list<type_pair<uchar3>,
                 type_pair<
#  if _CCCL_CTK_AT_LEAST(13, 0)
                   ulonglong4_16a
#  else // _CCCL_CTK_AT_LEAST(13, 0)
                   ulonglong4
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
                   >>;
#elif TEST_TYPES == 3
// clang-format off
using full_type_list = c2h::type_list<
type_pair<custom_t>
#if TEST_HALF_T()
, type_pair<half_t> // testing half
#endif // TEST_HALF_T()
#if TEST_BF_T()
, type_pair<bfloat16_t> // testing bf16
#endif // TEST_BF_T()
>;
// clang-format on
#endif

/**
 * @brief Input data generation mode
 */
enum class gen_data_t : int
{
  /// Uniform random data generation
  GEN_TYPE_RANDOM,
  /// Constant value as input data
  GEN_TYPE_CONST
};

struct VectorCompareResult
{
  std::vector<std::tuple<size_t, int, int>> first_mismatches;
  std::vector<std::tuple<size_t, int, int>> last_mismatches;
  size_t total_mismatches = 0;
  double mismatch_percent = 0.0;
  int max_difference      = 0;
};

VectorCompareResult compare_vectors(const std::vector<int>& actual, const std::vector<int>& expected)
{
  VectorCompareResult result;

  if (actual.size() != expected.size())
  {
    std::cerr << "Error: Vectors have different sizes (" << actual.size() << " vs " << expected.size() << ")\n";
    return result;
  }

  std::vector<std::tuple<size_t, int, int>> mismatches;
  mismatches.reserve(actual.size());
  int current_max_diff = 0;

  for (size_t i = 0; i < actual.size(); ++i)
  {
    if (actual[i] != expected[i])
    {
      mismatches.emplace_back(i, actual[i], expected[i]);
      current_max_diff = std::max(current_max_diff, std::abs(actual[i] - expected[i]));
    }
  }

  result.total_mismatches = mismatches.size();
  result.mismatch_percent = (static_cast<double>(result.total_mismatches) / actual.size()) * 100.0;
  result.max_difference   = current_max_diff;

  // Handle first 10 mismatches
  size_t first_count = std::min<size_t>(mismatches.size(), 10);
  result.first_mismatches.assign(mismatches.begin(), mismatches.begin() + first_count);

  // Handle last 10 mismatches
  if (mismatches.size() > 10)
  {
    auto start = mismatches.end() - std::min<size_t>(mismatches.size(), 10);
    result.last_mismatches.assign(start, mismatches.end());
  }
  else
  {
    result.last_mismatches = mismatches;
  }

  return result;
}

void print_comparison(const VectorCompareResult& res)
{
  // Print first mismatches
  std::cout << "First 10 mismatches:\n";
  for (const auto& [idx, a, b] : res.first_mismatches)
  {
    std::cout << "At index " << idx << ". Got " << a << ". Expected " << b << ". Difference " << a - b << "\n";
  }

  // Print last mismatches if different from first
  if (!res.last_mismatches.empty() && res.last_mismatches != res.first_mismatches)
  {
    std::cout << "\nLast 10 mismatches:\n";
    for (const auto& [idx, a, b] : res.last_mismatches)
    {
      std::cout << "At index " << idx << ". Got " << a << ". Expected " << b << ". Difference " << a - b << "\n";
    }
  }

  // Print summary
  std::cout
    << "\nTotal mismatches: " << res.total_mismatches << " (" << std::fixed << std::setprecision(2)
    << res.mismatch_percent << "%)\n"
    << "Maximum absolute difference: " << res.max_difference << "\n";
}

bool compareIsEqualAndPrint(const std::vector<int>& actual, const std::vector<int>& expected)
{
  VectorCompareResult result = compare_vectors(expected, actual);
  if (result.total_mismatches == 0)
  {
    return true;
  }
  else
  {
    print_comparison(result);
    return false;
  }
}

C2H_TEST("Device scan works with all device interfaces", "[scan][device]", full_type_list)
{
  using params   = params_t<TestType>;
  using input_t  = typename params::item_t;
  using output_t = typename params::output_t;
  using offset_t = int32_t;

  // constexpr offset_t min_items = 1;
  // constexpr offset_t max_items = 1000000;

  // Generate the input sizes to test for
  // const offset_t num_items = GENERATE_COPY(
  //   take(3, random(min_items, max_items)),
  //   values({
  //     min_items,
  //     max_items,
  //   }));

  SECTION("inclusive scan")
  {
    for (int ii = 2; ii < 1000; ii += 1)
    {
      const offset_t num_items = ii * 16;
      CAPTURE(num_items);
      // // Input data generation to test
      // const gen_data_t data_gen_mode = GENERATE_COPY(gen_data_t::GEN_TYPE_RANDOM, gen_data_t::GEN_TYPE_CONST);
      //
      using op_t    = cuda::std::plus<>;
      using accum_t = cuda::std::__accumulator_t<op_t, input_t, input_t>;

      const int max_offset = 16;

      // // Generate input data
      c2h::device_vector<input_t> in_items(num_items + max_offset, 42);
      c2h::gen(C2H_SEED(2), in_items);

      auto d_in_it = thrust::raw_pointer_cast(in_items.data());

      for (int offset = 0; offset < max_offset; ++offset)
      {
        CAPTURE(offset);
        // Prepare verification data
        c2h::host_vector<input_t> host_items(in_items);
        c2h::host_vector<output_t> expected_result(num_items);
        compute_inclusive_scan_reference(
          host_items.cbegin() + offset, host_items.cend() - max_offset + offset, expected_result.begin(), op_t{}, 0);

        // Run test
        c2h::device_vector<output_t> out_result(num_items);
        auto d_out_it = thrust::raw_pointer_cast(out_result.data());
        device_inclusive_scan(unwrap_it(d_in_it + offset), unwrap_it(d_out_it), op_t{}, num_items);

        // Verify result. Copy to a vector on the host for comparison.
        std::vector<output_t> expected_result_vec(expected_result.size());
        std::vector<output_t> out_result_vec(out_result.size());
        thrust::copy(expected_result.begin(), expected_result.end(), expected_result_vec.begin());
        thrust::copy(out_result.begin(), out_result.end(), out_result_vec.begin());

        REQUIRE(compareIsEqualAndPrint(expected_result_vec, out_result_vec));
        REQUIRE(expected_result == out_result);
      }
    }
  }
}
