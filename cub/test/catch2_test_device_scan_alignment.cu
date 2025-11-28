// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_scan.cuh>

#include <cuda/std/functional>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <vector>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_device_scan.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>

// %PARAM% TEST_LAUNCH lid 0

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScan, device_inclusive_scan);

// TODO(bgruber): the following functions implement better reporting for vector comparisons. We should generalize this.

template <typename T>
struct element_compare_result_t
{
  size_t index;
  T actual;
  T expected;
};

template <typename T>
struct vector_compare_result_t
{
  std::vector<element_compare_result_t<T>> first_mismatches;
  std::optional<std::vector<element_compare_result_t<T>>> last_mismatches;
  size_t total_mismatches = 0;
  double mismatch_percent = 0.0;
  T max_difference        = 0; // TODO(bgruber): we may want to reconsider this for a generic T
};

template <typename T>
vector_compare_result_t<T> compare_vectors(const c2h::host_vector<T>& actual, const c2h::host_vector<T>& expected)
{
  vector_compare_result_t<T> result;
  if (actual.size() != expected.size())
  {
    std::cerr << "Error: Vectors have different sizes (" << actual.size() << " vs " << expected.size() << ")\n";
    return result;
  }

  std::vector<element_compare_result_t<T>> mismatches;
  mismatches.reserve(actual.size()); // TODO(bgruber): this seems excessive
  T current_max_diff = 0;
  for (size_t i = 0; i < actual.size(); ++i)
  {
    if (actual[i] != expected[i])
    {
      mismatches.emplace_back(element_compare_result_t<T>{i, actual[i], expected[i]});
      T abs_diff       = actual[i] < expected[i] ? expected[i] - actual[i] : actual[i] - expected[i];
      current_max_diff = cuda::std::max(current_max_diff, abs_diff);
    }
  }

  result.total_mismatches = mismatches.size();
  result.mismatch_percent = (static_cast<double>(result.total_mismatches) / actual.size()) * 100.0;
  result.max_difference   = current_max_diff;

  // Handle first 10 mismatches
  size_t first_count = cuda::std::min<size_t>(mismatches.size(), 10);
  result.first_mismatches.assign(mismatches.begin(), mismatches.begin() + first_count);

  // Handle last 10 mismatches
  if (mismatches.size() > 10)
  {
    const auto start = mismatches.end() - cuda::std::min<size_t>(mismatches.size(), 10);
    result.last_mismatches.emplace(start, mismatches.end());
  }

  return result;
}

template <typename T>
void print_comparison(const vector_compare_result_t<T>& res)
{
  // Print first mismatches
  std::cout << "First 10 mismatches:\n";
  for (const auto& [idx, a, b] : res.first_mismatches)
  {
    std::cout << "At index " << idx << ". Got " << a << ". Expected " << b << ". Difference " << a - b << "\n";
  }

  // Print last mismatches if different from first
  if (res.last_mismatches)
  {
    std::cout << "\nLast 10 mismatches:\n";
    for (const auto& [idx, a, b] : *res.last_mismatches)
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

template <typename T>
bool compareIsEqualAndPrint(const c2h::host_vector<T>& actual, const c2h::host_vector<T>& expected)
{
  const vector_compare_result_t result = compare_vectors(expected, actual);
  if (result.total_mismatches == 0)
  {
    return true;
  }

  print_comparison(result);
  return false;
}

// TODO(bgruber): enable uint64, which exceeds the SMEM available on RTX 5090
// We cover types of various sizes smaller than 16 byte
using value_types = c2h::type_list<uint8_t, uint16_t, uint32_t /*, uint64_t*/>;

C2H_TEST("Device scan works with all device interfaces", "[scan][device]", value_types)
{
  using input_t  = c2h::get<0, TestType>;
  using output_t = input_t;
  using offset_t = int32_t;
  using op_t     = cuda::std::plus<>;

  constexpr int max_offset = 16;

  for (offset_t num_items = 2 * 16; num_items < 1000 * 16; num_items += 16)
  {
    CAPTURE(num_items);

    // Generate input data
    c2h::device_vector<input_t> in_items(num_items + max_offset, thrust::no_init);
    c2h::gen(C2H_SEED(1), in_items);
    auto d_in_it = thrust::raw_pointer_cast(in_items.data());

    // Prepare verification data
    c2h::host_vector<input_t> host_items(in_items);
    c2h::host_vector<output_t> expected_result(num_items, thrust::no_init);

    for (int offset = 0; offset < max_offset; ++offset)
    {
      CAPTURE(offset);

      // Compute verification data
      compute_inclusive_scan_reference(
        host_items.cbegin() + offset, host_items.cend() - max_offset + offset, expected_result.begin(), op_t{}, 0);

      // Run test
      c2h::device_vector<output_t> out_result(num_items, thrust::no_init);
      auto d_out_it = thrust::raw_pointer_cast(out_result.data());
      device_inclusive_scan(unwrap_it(d_in_it + offset), unwrap_it(d_out_it), op_t{}, num_items);

      REQUIRE(compareIsEqualAndPrint(expected_result, c2h::host_vector<output_t>(out_result)));
      REQUIRE(expected_result == out_result);
    }
  }
}
