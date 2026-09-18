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
#include <vector>

#include <cuda_runtime.h>

#include "algorithm_execution.h"
#include "test_util.h"
#include <cccl/c/scan_by_key.h>

using key_types = c2h::type_list<uint8_t, int16_t, uint32_t, int64_t>;

namespace
{
template <typename KeyT, typename ValueT>
std::vector<ValueT>
reference_scan_by_key(const std::vector<KeyT>& keys, const std::vector<ValueT>& values, bool inclusive, ValueT init)
{
  std::vector<ValueT> expected(values.size());
  ValueT running{};
  for (std::size_t i = 0; i < values.size(); ++i)
  {
    const bool run_start = (i == 0) || (keys[i] != keys[i - 1]);
    if (inclusive)
    {
      running     = run_start ? values[i] : static_cast<ValueT>(running + values[i]);
      expected[i] = running;
    }
    else
    {
      expected[i] = run_start ? init : running;
      running     = run_start ? static_cast<ValueT>(init + values[i]) : static_cast<ValueT>(running + values[i]);
    }
  }
  return expected;
}

template <typename KeyT, typename ValueT>
void check_scan_by_key(bool inclusive, std::size_t num_items, std::size_t run_length, ValueT init)
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
  pointer_t<ValueT> output_values_ptr(num_items);

  // A by-key launch links the scan and the equality operator into one module, but both helpers name
  // their entry point `op`, so the equality operator is renamed before it is compiled.
  std::string equality_src = get_unique_by_key_op(get_type_info<KeyT>().type);
  const std::string op_sym = "void op(";
  const auto op_sym_pos    = equality_src.find(op_sym);
  REQUIRE(op_sym_pos != std::string::npos);
  equality_src.replace(op_sym_pos, op_sym.size(), "void eq_op(");

  operation_t op          = make_operation("op", get_reduce_op(get_type_info<ValueT>().type));
  operation_t equality_op = make_operation("eq_op", equality_src);
  value_t<ValueT> init_storage{init};
  cccl_value_t init_value = init_storage;

  // The inclusive form has no init value in CUB, so it reports CCCL_NO_INIT and its kernel is the
  // inclusive one. The exclusive form seeds the head of every key run with `init`.
  const cccl_init_kind_t init_kind = inclusive ? CCCL_NO_INIT : CCCL_VALUE_INIT;

  const auto& build_info = BuildInformation<>::init();
  cccl_device_scan_by_key_build_result_t build{};
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_scan_by_key_build_ex(
      &build,
      input_keys_ptr,
      input_values_ptr,
      output_values_ptr,
      op,
      equality_op,
      init_value.type,
      inclusive,
      init_kind,
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
    == (inclusive ? cccl_device_inclusive_scan_by_key(
                      build, nullptr, &temp_storage_bytes, input_keys_ptr, input_values_ptr, output_values_ptr,
                      num_items, op, equality_op, CU_STREAM_LEGACY)
                  : cccl_device_exclusive_scan_by_key(
                      build, nullptr, &temp_storage_bytes, input_keys_ptr, input_values_ptr, output_values_ptr,
                      num_items, op, equality_op, init_value, CU_STREAM_LEGACY)));

  const pointer_t<char> temp_storage(temp_storage_bytes);
  REQUIRE(
    CUDA_SUCCESS
    == (inclusive ? cccl_device_inclusive_scan_by_key(
                      build, temp_storage.ptr, &temp_storage_bytes, input_keys_ptr, input_values_ptr, output_values_ptr,
                      num_items, op, equality_op, CU_STREAM_LEGACY)
                  : cccl_device_exclusive_scan_by_key(
                      build, temp_storage.ptr, &temp_storage_bytes, input_keys_ptr, input_values_ptr, output_values_ptr,
                      num_items, op, equality_op, init_value, CU_STREAM_LEGACY)));

  REQUIRE(std::vector<ValueT>(output_values_ptr) == reference_scan_by_key(input_keys, input_values, inclusive, init));
  REQUIRE(CUDA_SUCCESS == cccl_device_scan_by_key_cleanup(&build));
}
} // namespace

C2H_TEST("ScanByKey works", "[scan_by_key]", key_types)
{
  using key_t   = c2h::get<0, TestType>;
  using value_t = int32_t;

  const std::size_t num_items  = GENERATE(0, 1, 42, 1337, 42000);
  const std::size_t run_length = GENERATE(1, 3, 1000000); // all distinct, runs of three, all equal

  check_scan_by_key<key_t, value_t>(/*inclusive=*/true, num_items, run_length, value_t{0});
  check_scan_by_key<key_t, value_t>(/*inclusive=*/false, num_items, run_length, value_t{3});
}

// The tile state pairs the aggregate with an int run index, so an 8-byte accumulator still lands in
// the single-word form: 12 bytes plus the status byte rounds up to a 16-byte transaction word. The
// multi-word form needs an accumulator of 12 bytes or more, which this C API cannot reach today.
C2H_TEST("ScanByKey works with 64-bit keys and values", "[scan_by_key]")
{
  using key_t   = int64_t;
  using value_t = int64_t;

  const std::size_t num_items  = GENERATE(0, 1, 42, 1337, 42000);
  const std::size_t run_length = GENERATE(1, 3, 1000000);

  check_scan_by_key<key_t, value_t>(/*inclusive=*/true, num_items, run_length, value_t{0});
  check_scan_by_key<key_t, value_t>(/*inclusive=*/false, num_items, run_length, value_t{-7});
}
