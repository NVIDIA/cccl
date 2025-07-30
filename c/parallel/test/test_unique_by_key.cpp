//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <optional> // std::optional
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "algorithm_execution.h"
#include "build_result_caching.h"
#include "test_util.h"
#include <cccl/c/unique_by_key.h>

using key_types = c2h::type_list<uint8_t, int16_t, uint32_t, int64_t>;
using item_t    = int32_t;

using BuildResultT = cccl_device_unique_by_key_build_result_t;

struct unique_by_key_cleanup
{
  CUresult operator()(BuildResultT* build_data) const noexcept
  {
    return cccl_device_unique_by_key_cleanup(build_data);
  }
};

using unique_by_key_deleter       = BuildResultDeleter<BuildResultT, unique_by_key_cleanup>;
using unique_by_key_build_cache_t = build_cache_t<std::string, result_wrapper_t<BuildResultT, unique_by_key_deleter>>;

template <typename Tag>
auto& get_cache()
{
  return fixture<unique_by_key_build_cache_t, Tag>::get_or_create().get_value();
}

struct unique_by_key_build
{
  CUresult operator()(
    BuildResultT* build_ptr,
    cccl_iterator_t input_keys,
    cccl_iterator_t input_values,
    cccl_iterator_t output_keys,
    cccl_iterator_t output_values,
    cccl_iterator_t output_num_selected,
    cccl_op_t op,
    uint64_t,
    int cc_major,
    int cc_minor,
    const char* cub_path,
    const char* thrust_path,
    const char* libcudacxx_path,
    const char* ctk_path) const noexcept
  {
    return cccl_device_unique_by_key_build(
      build_ptr,
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
      ctk_path);
  }

  static bool should_check_sass(int cc_major)
  {
    // TODO: add a check for NVRTC version; ref nvbug 5243118
    return cc_major < 9;
  }
};

struct unique_by_key_run
{
  template <typename... Ts>
  CUresult operator()(Ts... args) const noexcept
  {
    return cccl_device_unique_by_key(args...);
  }
};

template <typename BuildCache = unique_by_key_build_cache_t, typename KeyT = std::string>
void unique_by_key(
  cccl_iterator_t input_keys,
  cccl_iterator_t input_values,
  cccl_iterator_t output_keys,
  cccl_iterator_t output_values,
  cccl_iterator_t output_num_selected,
  cccl_op_t op,
  uint64_t num_items,
  std::optional<BuildCache>& cache,
  const std::optional<KeyT>& lookup_key)
{
  AlgorithmExecute<BuildResultT, unique_by_key_build, unique_by_key_cleanup, unique_by_key_run, BuildCache, KeyT>(
    cache, lookup_key, input_keys, input_values, output_keys, output_values, output_num_selected, op, num_items);
}

// =============
//  Test section
// =============

struct UniqueByKey_AllPointerInputs_Fixture_Tag;
C2H_TEST("DeviceSelect::UniqueByKey can run with empty input", "[unique_by_key]", key_types)
{
  using key_t = c2h::get<0, TestType>;

  constexpr int num_items = 0;

  operation_t op = make_operation("op", get_unique_by_key_op(get_type_info<key_t>().type));
  std::vector<key_t> input_keys(num_items);

  pointer_t<key_t> input_keys_it(input_keys);
  pointer_t<int> output_num_selected_it(1);

  auto& input_items_it  = input_keys_it;
  auto& output_keys_it  = input_keys_it;
  auto& output_items_it = input_keys_it;

  auto& build_cache = get_cache<UniqueByKey_AllPointerInputs_Fixture_Tag>();
  // key: (input_type, output_type, num_selected_type)
  const auto& test_key = make_key<key_t, key_t, int>();

  unique_by_key(
    input_keys_it,
    input_items_it,
    output_keys_it,
    output_items_it,
    output_num_selected_it,
    op,
    num_items,
    build_cache,
    test_key);

  REQUIRE(0 == std::vector<int>(output_num_selected_it)[0]);
}

C2H_TEST("DeviceSelect::UniqueByKey works", "[unique_by_key]", key_types)
{
  using key_t = c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));

  operation_t op                   = make_operation("op", get_unique_by_key_op(get_type_info<key_t>().type));
  std::vector<key_t> input_keys    = generate<key_t>(num_items);
  std::vector<item_t> input_values = generate<item_t>(num_items);

  pointer_t<key_t> input_keys_it(input_keys);
  pointer_t<item_t> input_values_it(input_values);
  pointer_t<key_t> output_keys_it(num_items);
  pointer_t<item_t> output_values_it(num_items);
  pointer_t<int> output_num_selected_it(1);

  auto& build_cache = get_cache<UniqueByKey_AllPointerInputs_Fixture_Tag>();
  // key: (input_type, output_type, num_selected_type)
  const auto& test_key = make_key<key_t, item_t, int>();

  unique_by_key(
    input_keys_it,
    input_values_it,
    output_keys_it,
    output_values_it,
    output_num_selected_it,
    op,
    num_items,
    build_cache,
    test_key);

  std::vector<std::pair<key_t, item_t>> input_pairs;
  for (size_t i = 0; i < input_keys.size(); ++i)
  {
    input_pairs.emplace_back(input_keys[i], input_values[i]);
  }
  const auto boundary = std::unique(input_pairs.begin(), input_pairs.end(), [](const auto& a, const auto& b) {
    return a.first == b.first;
  });

  int num_selected = output_num_selected_it[0];

  REQUIRE((boundary - input_pairs.begin()) == num_selected);

  input_pairs.resize(num_selected);

  std::vector<key_t> host_output_keys(output_keys_it);
  std::vector<item_t> host_output_values(output_values_it);
  std::vector<std::pair<key_t, item_t>> output_pairs;
  for (int i = 0; i < num_selected; ++i)
  {
    output_pairs.emplace_back(host_output_keys[i], host_output_values[i]);
  }

  REQUIRE(input_pairs == output_pairs);
}

struct UniqueByKey_AllPointerInputs_WellKnown_Fixture_Tag;
C2H_TEST("DeviceSelect::UniqueByKey works with well-known operations", "[unique_by_key][well_known]", key_types)
{
  using key_t = c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));

  cccl_op_t op                     = make_well_known_unique_binary_predicate();
  std::vector<key_t> input_keys    = generate<key_t>(num_items);
  std::vector<item_t> input_values = generate<item_t>(num_items);

  pointer_t<key_t> input_keys_it(input_keys);
  pointer_t<item_t> input_values_it(input_values);
  pointer_t<key_t> output_keys_it(num_items);
  pointer_t<item_t> output_values_it(num_items);
  pointer_t<int> output_num_selected_it(1);

  auto& build_cache = get_cache<UniqueByKey_AllPointerInputs_WellKnown_Fixture_Tag>();
  // key: (input_type, output_type, num_selected_type)
  const auto& test_key = make_key<key_t, item_t, int>();

  unique_by_key(
    input_keys_it,
    input_values_it,
    output_keys_it,
    output_values_it,
    output_num_selected_it,
    op,
    num_items,
    build_cache,
    test_key);

  std::vector<std::pair<key_t, item_t>> input_pairs;
  for (size_t i = 0; i < input_keys.size(); ++i)
  {
    input_pairs.emplace_back(input_keys[i], input_values[i]);
  }
  const auto boundary = std::unique(input_pairs.begin(), input_pairs.end(), [](const auto& a, const auto& b) {
    return a.first == b.first;
  });

  int num_selected = output_num_selected_it[0];

  REQUIRE((boundary - input_pairs.begin()) == num_selected);

  input_pairs.resize(num_selected);

  std::vector<key_t> host_output_keys(output_keys_it);
  std::vector<item_t> host_output_values(output_values_it);
  std::vector<std::pair<key_t, item_t>> output_pairs;
  for (int i = 0; i < num_selected; ++i)
  {
    output_pairs.emplace_back(host_output_keys[i], host_output_values[i]);
  }

  REQUIRE(input_pairs == output_pairs);
}

C2H_TEST("DeviceSelect::UniqueByKey handles none equal", "[device][select_unique_by_key]", key_types)
{
  using key_t = c2h::get<0, TestType>;

  const int num_items = 250; // to ensure that we get none equal for smaller data types

  operation_t op                   = make_operation("op", get_unique_by_key_op(get_type_info<key_t>().type));
  std::vector<key_t> input_keys    = make_shuffled_sequence<key_t>(num_items);
  std::vector<item_t> input_values = generate<item_t>(num_items);

  pointer_t<key_t> input_keys_it(input_keys);
  pointer_t<item_t> input_values_it(input_values);
  pointer_t<key_t> output_keys_it(num_items);
  pointer_t<item_t> output_values_it(num_items);
  pointer_t<int> output_num_selected_it(1);

  auto& build_cache = get_cache<UniqueByKey_AllPointerInputs_Fixture_Tag>();
  // key: (input_type, output_type, num_selected_type)
  const auto& test_key = make_key<key_t, item_t, int>();

  unique_by_key(
    input_keys_it,
    input_values_it,
    output_keys_it,
    output_values_it,
    output_num_selected_it,
    op,
    num_items,
    build_cache,
    test_key);

  REQUIRE(num_items == std::vector<int>(output_num_selected_it)[0]);
  REQUIRE(input_keys == std::vector<key_t>(output_keys_it));
  REQUIRE(input_values == std::vector<item_t>(output_values_it));
}

C2H_TEST("DeviceSelect::UniqueByKey handles all equal", "[device][select_unique_by_key]", key_types)
{
  using key_t = c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));

  operation_t op = make_operation("op", get_unique_by_key_op(get_type_info<key_t>().type));
  std::vector<key_t> input_keys(num_items, static_cast<key_t>(1));
  std::vector<item_t> input_values = generate<item_t>(num_items);

  pointer_t<key_t> input_keys_it(input_keys);
  pointer_t<item_t> input_values_it(input_values);
  pointer_t<key_t> output_keys_it(1);
  pointer_t<item_t> output_values_it(1);
  pointer_t<int> output_num_selected_it(1);

  auto& build_cache = get_cache<UniqueByKey_AllPointerInputs_Fixture_Tag>();
  // key: (input_type, output_type, num_selected_type)
  const auto& test_key = make_key<key_t, item_t, int>();

  unique_by_key(
    input_keys_it,
    input_values_it,
    output_keys_it,
    output_values_it,
    output_num_selected_it,
    op,
    num_items,
    build_cache,
    test_key);

  REQUIRE(1 == std::vector<int>(output_num_selected_it)[0]);
  REQUIRE(input_keys[0] == std::vector<key_t>(output_keys_it)[0]);
  REQUIRE(input_values[0] == std::vector<item_t>(output_values_it)[0]);
}

struct key_pair
{
  short a;
  size_t b;

  bool operator==(const key_pair& other) const
  {
    return a == other.a && b == other.b;
  }
};

C2H_TEST("DeviceSelect::UniqueByKey works with custom types", "[device][select_unique_by_key]")
{
  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));

  operation_t op = make_operation(
    "op",
    "struct key_pair { short a; size_t b; };\n"
    "extern \"C\" __device__ void op(void* lhs_ptr, void* rhs_ptr, bool* out_ptr) {\n"
    "  key_pair* lhs = static_cast<key_pair*>(lhs_ptr);\n"
    "  key_pair* rhs = static_cast<key_pair*>(rhs_ptr);\n"
    "  bool* out = static_cast<bool*>(out_ptr);\n"
    "  *out = (lhs->a == rhs->a && lhs->b == rhs->b);\n"
    "}");
  const std::vector<short> a  = generate<short>(num_items);
  const std::vector<size_t> b = generate<size_t>(num_items);
  std::vector<key_pair> input_keys(num_items);
  std::vector<item_t> input_values = generate<item_t>(num_items);
  for (int i = 0; i < num_items; ++i)
  {
    input_keys[i] = key_pair{a[i], b[i]};
  }

  pointer_t<key_pair> input_keys_it(input_keys);
  pointer_t<item_t> input_values_it(input_values);
  pointer_t<key_pair> output_keys_it(num_items);
  pointer_t<item_t> output_values_it(num_items);
  pointer_t<int> output_num_selected_it(1);

  auto& build_cache = get_cache<UniqueByKey_AllPointerInputs_Fixture_Tag>();
  // key: (input_type, output_type, num_selected_type)
  const auto& test_key = make_key<key_pair, item_t, int>();

  unique_by_key(
    input_keys_it,
    input_values_it,
    output_keys_it,
    output_values_it,
    output_num_selected_it,
    op,
    num_items,
    build_cache,
    test_key);

  std::vector<std::pair<key_pair, item_t>> input_pairs;
  for (size_t i = 0; i < input_keys.size(); ++i)
  {
    input_pairs.emplace_back(input_keys[i], input_values[i]);
  }

  const auto boundary = std::unique(input_pairs.begin(), input_pairs.end(), [](const auto& a, const auto& b) {
    return a.first == b.first;
  });

  int num_selected = output_num_selected_it[0];

  REQUIRE((boundary - input_pairs.begin()) == num_selected);

  input_pairs.resize(num_selected);

  std::vector<key_pair> host_output_keys(output_keys_it);
  std::vector<item_t> host_output_values(output_values_it);
  std::vector<std::pair<key_pair, item_t>> output_pairs;
  for (int i = 0; i < num_selected; ++i)
  {
    output_pairs.emplace_back(host_output_keys[i], host_output_values[i]);
  }

  REQUIRE(input_pairs == output_pairs);
}

struct UniqueByKey_AllPointerInputs_WellKnown_Fixture_Tag;
C2H_TEST("DeviceSelect::UniqueByKey works with custom types with well-known operations",
         "[device][select_unique_by_key][well_known]")
{
  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));

  operation_t op_state = make_operation(
    "op",
    "struct key_pair { short a; size_t b; };\n"
    "extern \"C\" __device__ void op(void* lhs_ptr, void* rhs_ptr, bool* out_ptr) {\n"
    "  key_pair* lhs = static_cast<key_pair*>(lhs_ptr);\n"
    "  key_pair* rhs = static_cast<key_pair*>(rhs_ptr);\n"
    "  bool* out = static_cast<bool*>(out_ptr);\n"
    "  *out = (lhs->a == rhs->a && lhs->b == rhs->b);\n"
    "}");
  cccl_op_t op                = op_state;
  op.type                     = cccl_op_kind_t::CCCL_EQUAL_TO;
  const std::vector<short> a  = generate<short>(num_items);
  const std::vector<size_t> b = generate<size_t>(num_items);
  std::vector<key_pair> input_keys(num_items);
  std::vector<item_t> input_values = generate<item_t>(num_items);
  for (int i = 0; i < num_items; ++i)
  {
    input_keys[i] = key_pair{a[i], b[i]};
  }

  pointer_t<key_pair> input_keys_it(input_keys);
  pointer_t<item_t> input_values_it(input_values);
  pointer_t<key_pair> output_keys_it(num_items);
  pointer_t<item_t> output_values_it(num_items);
  pointer_t<int> output_num_selected_it(1);

  auto& build_cache = get_cache<UniqueByKey_AllPointerInputs_WellKnown_Fixture_Tag>();
  // key: (input_type, output_type, num_selected_type)
  const auto& test_key = make_key<key_pair, item_t, int>();

  unique_by_key(
    input_keys_it,
    input_values_it,
    output_keys_it,
    output_values_it,
    output_num_selected_it,
    op,
    num_items,
    build_cache,
    test_key);

  std::vector<std::pair<key_pair, item_t>> input_pairs;
  for (size_t i = 0; i < input_keys.size(); ++i)
  {
    input_pairs.emplace_back(input_keys[i], input_values[i]);
  }

  const auto boundary = std::unique(input_pairs.begin(), input_pairs.end(), [](const auto& a, const auto& b) {
    return a.first == b.first;
  });

  int num_selected = output_num_selected_it[0];

  REQUIRE((boundary - input_pairs.begin()) == num_selected);

  input_pairs.resize(num_selected);

  std::vector<key_pair> host_output_keys(output_keys_it);
  std::vector<item_t> host_output_values(output_values_it);
  std::vector<std::pair<key_pair, item_t>> output_pairs;
  for (int i = 0; i < num_selected; ++i)
  {
    output_pairs.emplace_back(host_output_keys[i], host_output_values[i]);
  }

  REQUIRE(input_pairs == output_pairs);
}

struct UniqueByKey_Iterators_Fixture_Tag;
C2H_TEST("DeviceMergeSort::SortPairs works with input and output iterators", "[merge_sort]")
{
  using T = int;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));

  operation_t op = make_operation("op", get_unique_by_key_op(get_type_info<int>().type));
  iterator_t<T, random_access_iterator_state_t<T>> input_keys_it =
    make_random_access_iterator<T>(iterator_kind::INPUT, "int", "key");
  iterator_t<T, random_access_iterator_state_t<T>> input_values_it =
    make_random_access_iterator<T>(iterator_kind::INPUT, "int", "value", " * 2");
  iterator_t<T, random_access_iterator_state_t<T>> output_keys_it =
    make_random_access_iterator<T>(iterator_kind::OUTPUT, "int", "key_out");
  iterator_t<T, random_access_iterator_state_t<T>> output_values_it =
    make_random_access_iterator<T>(iterator_kind::OUTPUT, "int", "value_out", " * 3");
  iterator_t<T, random_access_iterator_state_t<T>> output_num_selected_it =
    make_random_access_iterator<T>(iterator_kind::OUTPUT, "int", "num_selected");

  std::vector<T> input_keys        = generate<T>(num_items);
  std::vector<item_t> input_values = generate<int>(num_items);

  pointer_t<T> input_keys_ptr(input_keys);
  input_keys_it.state.data = input_keys_ptr.ptr;
  pointer_t<item_t> input_values_ptr(input_values);
  input_values_it.state.data = input_values_ptr.ptr;

  pointer_t<T> output_keys_ptr(num_items);
  output_keys_it.state.data = output_keys_ptr.ptr;
  pointer_t<item_t> output_values_ptr(num_items);
  output_values_it.state.data = output_values_ptr.ptr;

  pointer_t<int> output_num_selected_ptr(1);
  output_num_selected_it.state.data = output_num_selected_ptr.ptr;

  auto& build_cache = get_cache<UniqueByKey_Iterators_Fixture_Tag>();
  // key: (input_type, output_type, num_selected_type)
  const auto& test_key = make_key<T, T, int>();

  unique_by_key(
    input_keys_it,
    input_values_it,
    output_keys_it,
    output_values_it,
    output_num_selected_it,
    op,
    num_items,
    build_cache,
    test_key);

  std::vector<std::pair<T, item_t>> input_pairs;
  for (size_t i = 0; i < input_keys.size(); ++i)
  {
    // Multiplying by 6 since we multiply by 2 and 3 in the input and output value iterators
    input_pairs.emplace_back(input_keys[i], input_values[i] * 6);
  }
  const auto boundary = std::unique(input_pairs.begin(), input_pairs.end(), [](const auto& a, const auto& b) {
    return a.first == b.first;
  });

  int num_selected = output_num_selected_ptr[0];

  REQUIRE((boundary - input_pairs.begin()) == num_selected);

  input_pairs.resize(num_selected);

  std::vector<T> host_output_keys(output_keys_ptr);
  std::vector<item_t> host_output_values(output_values_ptr);
  std::vector<std::pair<T, item_t>> output_pairs;
  for (int i = 0; i < num_selected; ++i)
  {
    output_pairs.emplace_back(host_output_keys[i], host_output_values[i]);
  }

  REQUIRE(input_pairs == output_pairs);
}

struct large_key_pair
{
  int a;
  char c[100];

  bool operator==(const large_key_pair& other) const
  {
    return a == other.a;
  }
};

C2H_TEST("DeviceSelect::UniqueByKey fails to build for large types due to no vsmem", "[device][select_unique_by_key]")
{
  const int num_items = 1;

  operation_t op = make_operation(
    "op",
    "struct large_key_pair { int a; char c[100]; };\n"
    "extern \"C\" __device__ bool op(large_key_pair lhs, large_key_pair rhs) {\n"
    "  return lhs.a == rhs.a;\n"
    "}");
  const std::vector<int> a = generate<int>(num_items);
  std::vector<large_key_pair> input_keys(num_items);
  for (int i = 0; i < num_items; ++i)
  {
    input_keys[i] = large_key_pair{a[i], {}};
  }

  pointer_t<large_key_pair> input_keys_it(input_keys);
  pointer_t<item_t> input_values_it;
  pointer_t<large_key_pair> output_keys_it(num_items);
  pointer_t<item_t> output_values_it;
  pointer_t<int> output_num_selected_it(1);

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
    CUDA_ERROR_UNKNOWN
    == cccl_device_unique_by_key_build(
      &build,
      input_keys_it,
      input_values_it,
      output_keys_it,
      output_values_it,
      output_num_selected_it,
      op,
      cc_major,
      cc_minor,
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path));
}
