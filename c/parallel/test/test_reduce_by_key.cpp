//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "algorithm_execution.h"
#include "test_util.h"
#include <cccl/c/reduce_by_key.h>

using key_types = c2h::type_list<uint8_t, int16_t, uint32_t, int64_t>;

namespace
{
// ReduceByKey never sorts or groups: it walks the input once and collapses every run of adjacent equal
// keys into one output element. The surviving key of a run is its last one, exactly like
// `cub::DeviceReduce::ReduceByKey`.
template <typename KeyT, typename ValueT>
std::pair<std::vector<KeyT>, std::vector<ValueT>>
reference_reduce_by_key(const std::vector<KeyT>& keys, const std::vector<ValueT>& values)
{
  std::vector<KeyT> out_keys;
  std::vector<ValueT> out_values;
  for (std::size_t i = 0; i < keys.size(); ++i)
  {
    if (i == 0 || keys[i] != keys[i - 1])
    {
      out_keys.push_back(keys[i]);
      out_values.push_back(values[i]);
    }
    else
    {
      out_keys.back()   = keys[i];
      out_values.back() = static_cast<ValueT>(out_values.back() + values[i]);
    }
  }
  return {out_keys, out_values};
}

template <typename KeyT, typename ValueT>
void check_reduce_by_key(std::size_t num_items, std::size_t run_length)
{
  std::vector<KeyT> input_keys(num_items);
  std::vector<ValueT> input_values(num_items);
  for (std::size_t i = 0; i < num_items; ++i)
  {
    input_keys[i]   = static_cast<KeyT>(i / run_length);
    input_values[i] = static_cast<ValueT>(static_cast<int>(i % 7) - 3);
  }

  pointer_t<KeyT> input_keys_ptr(input_keys);
  pointer_t<ValueT> input_values_ptr(input_values);
  // The caller sizes both outputs for the worst case in which every key starts its own run; the C API
  // deliberately does not let the implementation assume anything smaller.
  pointer_t<KeyT> output_keys_ptr(num_items);
  pointer_t<ValueT> output_aggregates_ptr(num_items);
  pointer_t<int> num_runs_ptr(1);

  operation_t op = make_operation("op", get_reduce_op(get_type_info<ValueT>().type));

  const auto& build_info = BuildInformation<>::init();
  cccl_device_reduce_by_key_build_result_t build{};
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_reduce_by_key_build_ex(
      &build,
      input_keys_ptr,
      input_values_ptr,
      output_keys_ptr,
      output_aggregates_ptr,
      num_runs_ptr,
      op,
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path(),
      nullptr));

  std::size_t temp_storage_bytes = 0;
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_reduce_by_key(
      build,
      nullptr,
      &temp_storage_bytes,
      input_keys_ptr,
      input_values_ptr,
      output_keys_ptr,
      output_aggregates_ptr,
      num_runs_ptr,
      num_items,
      op,
      CU_STREAM_LEGACY));

  const pointer_t<char> temp_storage(temp_storage_bytes);
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_reduce_by_key(
      build,
      temp_storage.ptr,
      &temp_storage_bytes,
      input_keys_ptr,
      input_values_ptr,
      output_keys_ptr,
      output_aggregates_ptr,
      num_runs_ptr,
      num_items,
      op,
      CU_STREAM_LEGACY));

  const auto expected = reference_reduce_by_key(input_keys, input_values);
  const int num_runs  = std::vector<int>(num_runs_ptr).at(0);
  REQUIRE(num_runs == static_cast<int>(expected.first.size()));

  const std::vector<KeyT> actual_keys(output_keys_ptr);
  const std::vector<ValueT> actual_aggregates(output_aggregates_ptr);
  REQUIRE(std::vector<KeyT>(actual_keys.begin(), actual_keys.begin() + num_runs) == expected.first);
  REQUIRE(std::vector<ValueT>(actual_aggregates.begin(), actual_aggregates.begin() + num_runs) == expected.second);

  REQUIRE(CUDA_SUCCESS == cccl_device_reduce_by_key_cleanup(&build));
}
} // namespace

C2H_TEST("ReduceByKey works", "[reduce_by_key]", key_types)
{
  using key_t   = c2h::get<0, TestType>;
  using value_t = int32_t;

  const std::size_t num_items  = GENERATE(0, 1, 42, 1337, 42000);
  const std::size_t run_length = GENERATE(1, 3, 1000000); // all distinct, runs of three, all equal

  check_reduce_by_key<key_t, value_t>(num_items, run_length);
}

// A 64-bit accumulator pairs with the 32-bit run index into a 12 byte tile, which is still small enough
// for the single word tile state. Widening the accumulator past that is what switches over to the
// multi-word layout, and no type the C API accepts today is that large.
C2H_TEST("ReduceByKey works with 64-bit keys and values", "[reduce_by_key]")
{
  using key_t   = int64_t;
  using value_t = int64_t;

  const std::size_t num_items  = GENERATE(0, 1, 42, 1337, 42000);
  const std::size_t run_length = GENERATE(1, 3, 1000000);

  check_reduce_by_key<key_t, value_t>(num_items, run_length);
}
