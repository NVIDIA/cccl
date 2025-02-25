//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_runtime.h>

#include <vector>

#include "test_util.h"
#include <cccl/c/unique_by_key.h>

using key_types = std::tuple<uint8_t, int16_t, uint32_t, double>;
using item_t    = float;

void unique_by_key(
  cccl_iterator_t input_keys,
  cccl_iterator_t input_values,
  cccl_iterator_t output_keys,
  cccl_iterator_t output_values,
  cccl_iterator_t output_num_selected,
  cccl_op_t op,
  unsigned long long num_items)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  const int cc_major = deviceProp.major;
  const int cc_minor = deviceProp.minor;

  const char* cub_path        = TEST_CUB_PATH;
  const char* thrust_path     = TEST_THRUST_PATH;
  const char* libcudacxx_path = TEST_LIBCUDACXX_PATH;
  const char* ctk_path        = TEST_CTK_PATH;

  cccl_device_unique_by_key_build_result_t build;
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_unique_by_key_build(
      &build,
      input_keys,
      input_values,
      output_keys,
      output_values,
      output_num_selected,
      op,
      cc_major,
      cc_minor,
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path));

  const std::string sass = inspect_sass(build.cubin, build.cubin_size);
  REQUIRE(sass.find("LDL") == std::string::npos);
  REQUIRE(sass.find("STL") == std::string::npos);

  size_t temp_storage_bytes = 0;
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_unique_by_key(
      build,
      nullptr,
      &temp_storage_bytes,
      input_keys,
      input_values,
      output_keys,
      output_values,
      output_num_selected,
      op,
      num_items,
      0));

  pointer_t<uint8_t> temp_storage(temp_storage_bytes);

  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_unique_by_key(
      build,
      temp_storage.ptr,
      &temp_storage_bytes,
      input_keys,
      input_values,
      output_keys,
      output_values,
      output_num_selected,
      op,
      num_items,
      0));
  REQUIRE(CUDA_SUCCESS == cccl_device_unique_by_key_cleanup(&build));
}

TEMPLATE_LIST_TEST_CASE("DeviceSelect::UniqueByKey can run with empty input", "[unique_by_key]", key_types)
{
  constexpr int num_items = 0;

  operation_t op = make_operation("op", get_unique_by_key_op(get_type_info<TestType>().type));
  std::vector<TestType> input_keys(num_items);
  std::vector<int> output_num_selected(1, 0);

  pointer_t<TestType> input_keys_it(input_keys);
  pointer_t<int> output_num_selected_it(output_num_selected);

  unique_by_key(input_keys_it, input_keys_it, input_keys_it, input_keys_it, output_num_selected_it, op, num_items);

  REQUIRE(0 == std::vector<int>(output_num_selected_it)[0]);
}
