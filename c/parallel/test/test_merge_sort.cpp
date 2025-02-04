//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "test_util.h"
#include <cccl/c/merge_sort.h>

using key_types = std::tuple<std::uint8_t, std::int16_t, std::uint32_t, double>;

void merge_sort(cccl_iterator_t input_keys,
                cccl_iterator_t input_items,
                cccl_iterator_t output_keys,
                cccl_iterator_t output_items,
                unsigned long long num_items,
                cccl_op_t op)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  const int cc_major = deviceProp.major;
  const int cc_minor = deviceProp.minor;

  const char* cub_path        = TEST_CUB_PATH;
  const char* thrust_path     = TEST_THRUST_PATH;
  const char* libcudacxx_path = TEST_LIBCUDACXX_PATH;
  const char* ctk_path        = TEST_CTK_PATH;

  cccl_device_merge_sort_build_result_t build;
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_merge_sort_build(
      &build,
      input_keys,
      input_items,
      output_keys,
      output_items,
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
  REQUIRE(CUDA_SUCCESS
          == cccl_device_merge_sort(
            build, nullptr, &temp_storage_bytes, input_keys, input_items, output_keys, output_items, num_items, op, 0));

  pointer_t<uint8_t> temp_storage(temp_storage_bytes);

  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_merge_sort(
      build, temp_storage.ptr, &temp_storage_bytes, input_keys, input_items, output_keys, output_items, num_items, op, 0));
  REQUIRE(CUDA_SUCCESS == cccl_device_merge_sort_cleanup(&build));
}

TEMPLATE_LIST_TEST_CASE("DeviceMergeSort::SortKeys works", "[merge_sort]", key_types)
{
  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));

  operation_t op                      = make_operation("op", get_merge_sort_op(get_type_info<TestType>().type));
  std::vector<TestType> input_keys    = make_shuffled_key_ranks_vector<TestType>(num_items);
  std::vector<TestType> expected_keys = input_keys;

  pointer_t<TestType> input_keys_it(input_keys);
  pointer_t<TestType> input_items_it;
  pointer_t<TestType> output_keys_it(input_keys);
  pointer_t<TestType> output_items_it;

  merge_sort(input_keys_it, input_items_it, output_keys_it, output_items_it, num_items, op);

  std::sort(expected_keys.begin(), expected_keys.end());
  REQUIRE(expected_keys == std::vector<TestType>(output_keys_it));
}

TEMPLATE_LIST_TEST_CASE("DeviceMergeSort::SortKeysCopy works", "[merge_sort]", key_types)
{
  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));

  operation_t op                   = make_operation("op", get_merge_sort_op(get_type_info<TestType>().type));
  std::vector<TestType> input_keys = make_shuffled_key_ranks_vector<TestType>(num_items);
  std::vector<TestType> output_keys(num_items);
  std::vector<TestType> expected_keys = input_keys;

  pointer_t<TestType> input_keys_it(input_keys);
  pointer_t<TestType> input_items_it;
  pointer_t<TestType> output_keys_it(output_keys);
  pointer_t<TestType> output_items_it;

  merge_sort(input_keys_it, input_items_it, output_keys_it, output_items_it, num_items, op);

  std::sort(expected_keys.begin(), expected_keys.end());
  REQUIRE(expected_keys == std::vector<TestType>(output_keys_it));
}
