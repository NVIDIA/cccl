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

#include <cstdint>
#include <iostream> // std::cerr
#include <optional> // std::optional
#include <string>

#include "build_result_caching.h"
#include "test_util.h"
#include <cccl/c/radix_sort.h>

using key_types = std::tuple<uint8_t, int16_t, uint32_t, double>;
using item_t    = float;

struct radix_sort_build_cleaner
{
  void operator()(cccl_device_radix_sort_build_result_t* build_data) noexcept
  {
    auto command_status = cccl_device_radix_sort_cleanup(build_data);
    if (CUDA_SUCCESS != command_status)
    {
      std::cerr << "  Clean-up call returned status " << command_status << ". The pointer was "
                << static_cast<void*>(build_data) << std::endl;
      if (build_data)
      {
        std::cerr << "build->cc: " << build_data->cc << ", build->cubin: " << build_data->cubin
                  << ", build->cubin_size: " << build_data->cubin_size << ", order: " << build_data->order << std::endl;
      }
    };
  }
};

using radix_sort_build_cache_t =
  build_cache_t<std::string, result_wrapper_t<cccl_device_radix_sort_build_result_t, radix_sort_build_cleaner>>;

template <typename BuildCache = radix_sort_build_cache_t, typename KeyT = std::string>
void radix_sort(
  cccl_sort_order_t sort_order,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_keys_out,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  cccl_op_t decomposer,
  const char* decomposer_return_type,
  uint64_t num_items,
  int begin_bit,
  int end_bit,
  bool is_overwrite_okay,
  int* selector,
  std::optional<BuildCache>& cache,
  const std::optional<KeyT>& lookup_key)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  const int cc_major = deviceProp.major;
  const int cc_minor = deviceProp.minor;

  const char* cub_path        = TEST_CUB_PATH;
  const char* thrust_path     = TEST_THRUST_PATH;
  const char* libcudacxx_path = TEST_LIBCUDACXX_PATH;
  const char* ctk_path        = TEST_CTK_PATH;

  cccl_device_radix_sort_build_result_t build;
  bool found = false;

  const bool cache_and_key = bool(cache) && bool(lookup_key);

  if (cache_and_key)
  {
    auto& cache_v     = cache.value();
    const auto& key_v = lookup_key.value();
    if (cache_v.contains(key_v))
    {
      build = cache_v.get(key_v).get();
      found = true;
    }
  }

  if (!found)
  {
    REQUIRE(
      CUDA_SUCCESS
      == cccl_device_radix_sort_build(
        &build,
        sort_order,
        d_keys_in,
        d_values_in,
        decomposer,
        decomposer_return_type,
        cc_major,
        cc_minor,
        cub_path,
        thrust_path,
        libcudacxx_path,
        ctk_path));

    if (cache_and_key)
    {
      auto& cache_v     = cache.value();
      const auto& key_v = lookup_key.value();
      cache_v.insert(key_v, build);
    }
  }

  const std::string sass = inspect_sass(build.cubin, build.cubin_size);
  REQUIRE(sass.find("LDL") == std::string::npos);
  REQUIRE(sass.find("STL") == std::string::npos);

  size_t temp_storage_bytes = 0;
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_radix_sort(
      build,
      nullptr,
      &temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      decomposer,
      num_items,
      begin_bit,
      end_bit,
      is_overwrite_okay,
      selector,
      0));

  pointer_t<uint8_t> temp_storage(temp_storage_bytes);

  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_radix_sort(
      build,
      temp_storage.ptr,
      &temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      decomposer,
      num_items,
      begin_bit,
      end_bit,
      is_overwrite_okay,
      selector,
      0));

  if (cache_and_key)
  {
    // if cache and lookup_key were provided, the ownership of resources
    // allocated for build is transferred to the cache
  }
  else
  {
    // release build data resources
    REQUIRE(CUDA_SUCCESS == cccl_device_radix_sort_cleanup(&build));
  }
}

struct DeviceRadixSort_SortKeys_Fixture_Tag;
TEMPLATE_LIST_TEST_CASE("DeviceRadixSort::SortKeys works", "[radix_sort]", key_types)
{
  // We want a mix of small and large sizes because different implementations will be called
  const int num_items      = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));
  const bool is_descending = GENERATE(false, true);
  const auto order         = is_descending ? CCCL_DESCENDING : CCCL_ASCENDING;

  const int begin_bit          = 0;
  const int end_bit            = sizeof(TestType) * 8;
  const bool is_overwrite_okay = GENERATE(false, true);
  int selector                 = -1;

  static constexpr cccl_op_t decomposer_no_op{};
  static constexpr const char* unused_decomposer_retty = "";

  // problem descriptor: (order, TestType, item_t, is_overwrite_ok, items_present = false)
  std::vector<TestType> input_keys    = make_shuffled_sequence<TestType>(num_items);
  std::vector<TestType> expected_keys = input_keys;

  pointer_t<TestType> input_keys_it(input_keys);
  pointer_t<TestType> output_keys_it(num_items);

  pointer_t<item_t> input_items_it, output_items_it;

  auto& build_cache =
    fixture<radix_sort_build_cache_t, DeviceRadixSort_SortKeys_Fixture_Tag>::get_or_create().get_value();

  std::string key_string = KeyBuilder::join(
    {KeyBuilder::bool_as_key(is_descending),
     KeyBuilder::type_as_key<TestType>(),
     KeyBuilder::type_as_key<item_t>(),
     KeyBuilder::bool_as_key(is_overwrite_okay)});
  std::optional<std::string> test_key{key_string};

  radix_sort(
    order,
    input_keys_it,
    output_keys_it,
    input_items_it,
    output_items_it,
    decomposer_no_op,
    unused_decomposer_retty,
    num_items,
    begin_bit,
    end_bit,
    is_overwrite_okay,
    &selector,
    build_cache,
    test_key);

  assert(selector == 0 || selector == 1);

  if (is_descending)
  {
    std::sort(expected_keys.begin(), expected_keys.end(), std::greater<TestType>());
  }
  else
  {
    std::sort(expected_keys.begin(), expected_keys.end());
  }

  auto& output_keys = (is_overwrite_okay && selector == 0) ? input_keys_it : output_keys_it;
  REQUIRE(expected_keys == std::vector<TestType>(output_keys));
}

struct DeviceRadixSort_SortPairs_Fixture_Tag;
TEMPLATE_LIST_TEST_CASE("DeviceRadixSort::SortPairs works", "[radix_sort]", key_types)
{
  const int num_items      = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));
  const bool is_descending = GENERATE(false, true);
  const auto order         = is_descending ? CCCL_DESCENDING : CCCL_ASCENDING;

  const int begin_bit          = 0;
  const int end_bit            = sizeof(TestType) * 8;
  const bool is_overwrite_okay = GENERATE(false, true);
  int selector                 = -1;

  static constexpr cccl_op_t decomposer_no_op{};
  static constexpr const char* unused_decomposer_retty = "";

  // problem descriptor in this example: (order, TestType, item_t, is_overwrite_ok)

  std::vector<TestType> input_keys = make_shuffled_sequence<TestType>(num_items);
  std::vector<item_t> input_items(num_items);
  std::transform(input_keys.begin(), input_keys.end(), input_items.begin(), [](TestType key) {
    return static_cast<item_t>(key);
  });

  std::vector<TestType> expected_keys = input_keys;
  std::vector<item_t> expected_items  = input_items;

  pointer_t<TestType> input_keys_it(input_keys);
  pointer_t<TestType> output_keys_it(num_items);

  pointer_t<item_t> input_items_it(input_items);
  pointer_t<item_t> output_items_it(num_items);

  auto& build_cache =
    fixture<radix_sort_build_cache_t, DeviceRadixSort_SortPairs_Fixture_Tag>::get_or_create().get_value();

  std::string key_string = KeyBuilder::join(
    {KeyBuilder::bool_as_key(is_descending),
     KeyBuilder::type_as_key<TestType>(),
     KeyBuilder::type_as_key<item_t>(),
     KeyBuilder::bool_as_key(is_overwrite_okay)});

  std::optional<std::string> test_key{key_string};

  radix_sort(
    order,
    input_keys_it,
    output_keys_it,
    input_items_it,
    output_items_it,
    decomposer_no_op,
    unused_decomposer_retty,
    num_items,
    begin_bit,
    end_bit,
    is_overwrite_okay,
    &selector,
    build_cache,
    test_key);

  assert(selector == 0 || selector == 1);

  if (is_descending)
  {
    std::sort(expected_keys.begin(), expected_keys.end(), std::greater<TestType>());
    std::sort(expected_items.begin(), expected_items.end(), std::greater<TestType>());
  }
  else
  {
    std::sort(expected_keys.begin(), expected_keys.end());
    std::sort(expected_items.begin(), expected_items.end());
  }

  auto& output_keys  = (is_overwrite_okay && selector == 0) ? input_keys_it : output_keys_it;
  auto& output_items = (is_overwrite_okay && selector == 0) ? input_items_it : output_items_it;
  REQUIRE(expected_keys == std::vector<TestType>(output_keys));
  REQUIRE(expected_items == std::vector<item_t>(output_items));
}
