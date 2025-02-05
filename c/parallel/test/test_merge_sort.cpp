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

#include "test_util.h"
#include <cccl/c/merge_sort.h>

using key_types = std::tuple<uint8_t, int16_t, uint32_t, double>;
using item_t    = float;

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

  merge_sort(input_keys_it, input_items_it, input_keys_it, input_items_it, num_items, op);

  std::sort(expected_keys.begin(), expected_keys.end());
  REQUIRE(expected_keys == std::vector<TestType>(input_keys_it));
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

  merge_sort(input_keys_it, input_items_it, output_keys_it, input_items_it, num_items, op);

  std::sort(expected_keys.begin(), expected_keys.end());
  REQUIRE(expected_keys == std::vector<TestType>(output_keys_it));
}

TEMPLATE_LIST_TEST_CASE("DeviceMergeSort::SortPairs works", "[merge_sort]", key_types)
{
  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));

  operation_t op                   = make_operation("op", get_merge_sort_op(get_type_info<TestType>().type));
  std::vector<TestType> input_keys = make_shuffled_key_ranks_vector<TestType>(num_items);
  std::vector<item_t> input_items(num_items);
  std::transform(input_keys.begin(), input_keys.end(), input_items.begin(), [](TestType key) {
    return static_cast<item_t>(key);
  });
  std::vector<TestType> expected_keys = input_keys;
  std::vector<item_t> expected_items  = input_items;

  pointer_t<TestType> input_keys_it(input_keys);
  pointer_t<item_t> input_items_it(input_items);

  merge_sort(input_keys_it, input_items_it, input_keys_it, input_items_it, num_items, op);

  std::sort(expected_keys.begin(), expected_keys.end());
  std::sort(expected_items.begin(), expected_items.end());
  REQUIRE(expected_keys == std::vector<TestType>(input_keys_it));
  REQUIRE(expected_items == std::vector<item_t>(input_items_it));
}

TEMPLATE_LIST_TEST_CASE("DeviceMergeSort::SortPairsCopy works ", "[merge_sort]", key_types)
{
  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));

  operation_t op                   = make_operation("op", get_merge_sort_op(get_type_info<TestType>().type));
  std::vector<TestType> input_keys = make_shuffled_key_ranks_vector<TestType>(num_items);
  std::vector<item_t> input_items(num_items);
  std::transform(input_keys.begin(), input_keys.end(), input_items.begin(), [](TestType key) {
    return static_cast<item_t>(key);
  });
  std::vector<TestType> output_keys(num_items);
  std::vector<item_t> output_items(num_items);
  std::vector<TestType> expected_keys = input_keys;
  std::vector<item_t> expected_items  = input_items;

  pointer_t<TestType> input_keys_it(input_keys);
  pointer_t<item_t> input_items_it(input_items);
  pointer_t<TestType> output_keys_it(output_keys);
  pointer_t<item_t> output_items_it(output_items);

  merge_sort(input_keys_it, input_items_it, output_keys_it, output_items_it, num_items, op);

  std::sort(expected_keys.begin(), expected_keys.end());
  std::sort(expected_items.begin(), expected_items.end());
  REQUIRE(expected_keys == std::vector<TestType>(output_keys_it));
  REQUIRE(expected_items == std::vector<item_t>(output_items_it));
}

struct key_pair
{
  short a;
  size_t b;
};

struct item_pair
{
  int a;
  float b;
};

TEST_CASE("DeviceMergeSort:SortPairsCopy works with custom types", "[merge_sort]")
{
  const size_t num_items = GENERATE_COPY(take(2, random(1, 10)), values({5, 100, 200}));
  operation_t op         = make_operation(
    "op",
    "struct key_pair { short a; size_t b; };\n"
            "extern \"C\" __device__ bool op(key_pair lhs, key_pair rhs) {\n"
            "  return lhs.a == rhs.a ? lhs.b < rhs.b : lhs.a < rhs.a;\n"
            "}");
  const std::vector<short> a  = generate<short>(num_items);
  const std::vector<size_t> b = generate<size_t>(num_items);
  std::vector<key_pair> input_keys(num_items);
  std::vector<item_pair> input_items(num_items);
  for (std::size_t i = 0; i < num_items; ++i)
  {
    input_keys[i]  = key_pair{a[i], b[i]};
    input_items[i] = item_pair{static_cast<int>(a[i]), static_cast<float>(b[i])};
  }
  std::vector<key_pair> expected_keys   = input_keys;
  std::vector<item_pair> expected_items = input_items;

  pointer_t<key_pair> input_keys_it(input_keys);
  pointer_t<item_pair> input_items_it(input_items);
  pointer_t<key_pair> output_keys_it(input_keys);
  pointer_t<item_pair> output_items_it(input_items);

  merge_sort(input_keys_it, input_items_it, output_keys_it, output_items_it, num_items, op);

  std::sort(expected_keys.begin(), expected_keys.end(), [](const key_pair& lhs, const key_pair& rhs) {
    return lhs.a == rhs.a ? lhs.b < rhs.b : lhs.a < rhs.a;
  });
  std::sort(expected_items.begin(), expected_items.end(), [](const item_pair& lhs, const item_pair& rhs) {
    return lhs.a == rhs.a ? lhs.b < rhs.b : lhs.a < rhs.a;
  });
  REQUIRE(std::equal(
    expected_keys.begin(),
    expected_keys.end(),
    std::vector<key_pair>(output_keys_it).begin(),
    [](const key_pair& lhs, const key_pair& rhs) {
      return lhs.a == rhs.a && lhs.b == rhs.b;
    }));
  REQUIRE(std::equal(
    expected_items.begin(),
    expected_items.end(),
    std::vector<item_pair>(output_items_it).begin(),
    [](const item_pair& lhs, const item_pair& rhs) {
      return lhs.a == rhs.a && lhs.b == rhs.b;
    }));
}

struct random_access_iterator_state_t
{
  int* d_input;
  size_t index;
};

TEST_CASE("DeviceMergeSort:SortKeys works with input iterators", "[merge_sort]")
{
  using TestType      = int;
  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));

  operation_t op = make_operation("op", get_merge_sort_op(get_type_info<TestType>().type));
  iterator_t<TestType, random_access_iterator_state_t> input_keys_it =
    make_iterator<TestType, random_access_iterator_state_t>(
      "struct random_access_iterator_state_t { int* d_input; size_t index; };\n",
      {"advance",
       "extern \"C\" __device__ void advance(random_access_iterator_state_t* state, unsigned long long offset) {\n"
       "  state->index += offset;\n"
       "}"},
      {"dereference",
       "extern \"C\" __device__ int dereference(random_access_iterator_state_t* state) {\n"
       "  return state->d_input[state->index];\n"
       "}"});
  std::vector<TestType> input_keys    = make_shuffled_key_ranks_vector<TestType>(num_items);
  std::vector<TestType> expected_keys = input_keys;

  pointer_t<TestType> input_keys_ptr(input_keys);
  input_keys_it.state.d_input = input_keys_ptr.ptr;
  input_keys_it.state.index   = 0;
  pointer_t<TestType> input_items_it;
  pointer_t<TestType> output_items_it;

  merge_sort(input_keys_it, input_items_it, input_keys_ptr, output_items_it, num_items, op);

  std::sort(expected_keys.begin(), expected_keys.end());
  REQUIRE(expected_keys == std::vector<TestType>(input_keys_ptr));
}
