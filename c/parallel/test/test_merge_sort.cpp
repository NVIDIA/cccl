//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>

#include <cuda_runtime.h>

#include "algorithm_execution.h"
#include "build_result_caching.h"
#include "test_util.h"
#include <cccl/c/merge_sort.h>

using key_types =
  c2h::type_list<uint8_t,
                 int16_t,
                 uint32_t,
#if _CCCL_HAS_NVFP16()
                 __half,
#endif
                 double>;
using item_t = float;

using BuildResultT = cccl_device_merge_sort_build_result_t;

struct merge_sort_cleanup
{
  CUresult operator()(BuildResultT* build_data) const noexcept
  {
    return cccl_device_merge_sort_cleanup(build_data);
  }
};

using merge_sort_deleter       = BuildResultDeleter<BuildResultT, merge_sort_cleanup>;
using merge_sort_build_cache_t = build_cache_t<std::string, result_wrapper_t<BuildResultT, merge_sort_deleter>>;

template <typename Tag>
auto& get_cache()
{
  return fixture<merge_sort_build_cache_t, Tag>::get_or_create().get_value();
}

struct merge_sort_build
{
  template <typename... Rest>
  CUresult operator()(
    BuildResultT* build_ptr,
    cccl_iterator_t input_keys,
    cccl_iterator_t input_items,
    cccl_iterator_t output_keys,
    cccl_iterator_t output_items,
    uint64_t,
    cccl_op_t op,
    Rest... rest) const noexcept
  {
    return cccl_device_merge_sort_build(build_ptr, input_keys, input_items, output_keys, output_items, op, rest...);
  }
};

struct merge_sort_run
{
  template <typename... Args>
  CUresult operator()(Args... args) const noexcept
  {
    return cccl_device_merge_sort(args...);
  }
};

template <typename BuildCache = merge_sort_build_cache_t, typename KeyT = std::string>
void merge_sort(
  cccl_iterator_t input_keys,
  cccl_iterator_t input_items,
  cccl_iterator_t output_keys,
  cccl_iterator_t output_items,
  uint64_t num_items,
  cccl_op_t op,
  std::optional<BuildCache>& cache,
  const std::optional<KeyT>& lookup_key)
{
  AlgorithmExecute<BuildResultT, merge_sort_build, merge_sort_cleanup, merge_sort_run, BuildCache, KeyT>(
    cache, lookup_key, input_keys, input_items, output_keys, output_items, num_items, op);
}

// ================
//   Start of tests
// ================

struct DeviceMergeSort_SortKeys_Fixture_Tag;
C2H_TEST("DeviceMergeSort::SortKeys works", "[merge_sort]", key_types)
{
  using key_t = c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));

  operation_t op                   = make_operation("op", get_merge_sort_op(get_type_info<key_t>().type));
  std::vector<key_t> input_keys    = make_shuffled_sequence<key_t>(num_items);
  std::vector<key_t> expected_keys = input_keys;

  pointer_t<key_t> input_keys_it(input_keys);
  pointer_t<key_t> input_items_it;

  auto& build_cache    = get_cache<DeviceMergeSort_SortKeys_Fixture_Tag>();
  const auto& test_key = make_key<key_t>();

  merge_sort(input_keys_it, input_items_it, input_keys_it, input_items_it, num_items, op, build_cache, test_key);

  std::sort(expected_keys.begin(), expected_keys.end());
  REQUIRE(expected_keys == std::vector<key_t>(input_keys_it));
}

struct DeviceMergeSort_SortKeys_WellKnown_Fixture_Tag;
C2H_TEST("DeviceMergeSort::SortKeys works with well-known predicate", "[merge_sort][well_known]", key_types)
{
  using key_t = c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));

  cccl_op_t op                     = make_well_known_binary_predicate();
  std::vector<key_t> input_keys    = make_shuffled_sequence<key_t>(num_items);
  std::vector<key_t> expected_keys = input_keys;

  pointer_t<key_t> input_keys_it(input_keys);
  pointer_t<key_t> input_items_it;

  auto& build_cache    = get_cache<DeviceMergeSort_SortKeys_WellKnown_Fixture_Tag>();
  const auto& test_key = make_key<key_t>();

  merge_sort(input_keys_it, input_items_it, input_keys_it, input_items_it, num_items, op, build_cache, test_key);

  std::sort(expected_keys.begin(), expected_keys.end());
  REQUIRE(expected_keys == std::vector<key_t>(input_keys_it));
}

struct DeviceMergeSort_SortKeysCopy_Fixture_Tag;
C2H_TEST("DeviceMergeSort::SortKeysCopy works", "[merge_sort]", key_types)
{
  using key_t = c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));

  operation_t op                = make_operation("op", get_merge_sort_op(get_type_info<key_t>().type));
  std::vector<key_t> input_keys = make_shuffled_sequence<key_t>(num_items);
  std::vector<key_t> output_keys(num_items);
  std::vector<key_t> expected_keys = input_keys;

  pointer_t<key_t> input_keys_it(input_keys);
  pointer_t<key_t> input_items_it;
  pointer_t<key_t> output_keys_it(output_keys);

  auto& build_cache    = get_cache<DeviceMergeSort_SortKeysCopy_Fixture_Tag>();
  const auto& test_key = make_key<key_t>();

  merge_sort(input_keys_it, input_items_it, output_keys_it, input_items_it, num_items, op, build_cache, test_key);

  std::sort(expected_keys.begin(), expected_keys.end());
  REQUIRE(expected_keys == std::vector<key_t>(output_keys_it));
}

struct DeviceMergeSort_SortPairs_Fixture_Tag;
C2H_TEST("DeviceMergeSort::SortPairs works", "[merge_sort]", key_types)
{
  using key_t = c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));

  operation_t op                = make_operation("op", get_merge_sort_op(get_type_info<key_t>().type));
  std::vector<key_t> input_keys = make_shuffled_sequence<key_t>(num_items);
  std::vector<item_t> input_items(num_items);
  std::transform(input_keys.begin(), input_keys.end(), input_items.begin(), [](key_t key) {
    return static_cast<item_t>(key);
  });
  std::vector<key_t> expected_keys   = input_keys;
  std::vector<item_t> expected_items = input_items;

  pointer_t<key_t> input_keys_it(input_keys);
  pointer_t<item_t> input_items_it(input_items);

  auto& build_cache    = get_cache<DeviceMergeSort_SortPairs_Fixture_Tag>();
  const auto& test_key = make_key<key_t, item_t>();

  merge_sort(input_keys_it, input_items_it, input_keys_it, input_items_it, num_items, op, build_cache, test_key);

  std::sort(expected_keys.begin(), expected_keys.end());
  std::sort(expected_items.begin(), expected_items.end());
  REQUIRE(expected_keys == std::vector<key_t>(input_keys_it));
  REQUIRE(expected_items == std::vector<item_t>(input_items_it));
}

struct DeviceMergeSort_SortPairsCopy_Fixture_Tag;
C2H_TEST("DeviceMergeSort::SortPairsCopy works ", "[merge_sort]", key_types)
{
  using key_t = c2h::get<0, TestType>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));

  operation_t op                = make_operation("op", get_merge_sort_op(get_type_info<key_t>().type));
  std::vector<key_t> input_keys = make_shuffled_sequence<key_t>(num_items);
  std::vector<item_t> input_items(num_items);
  std::transform(input_keys.begin(), input_keys.end(), input_items.begin(), [](key_t key) {
    return static_cast<item_t>(key);
  });
  std::vector<key_t> output_keys(num_items);
  std::vector<item_t> output_items(num_items);
  std::vector<key_t> expected_keys   = input_keys;
  std::vector<item_t> expected_items = input_items;

  pointer_t<key_t> input_keys_it(input_keys);
  pointer_t<item_t> input_items_it(input_items);
  pointer_t<key_t> output_keys_it(output_keys);
  pointer_t<item_t> output_items_it(output_items);

  auto& build_cache    = get_cache<DeviceMergeSort_SortPairs_Fixture_Tag>();
  const auto& test_key = make_key<key_t, item_t>();

  merge_sort(input_keys_it, input_items_it, output_keys_it, output_items_it, num_items, op, build_cache, test_key);

  std::sort(expected_keys.begin(), expected_keys.end());
  std::sort(expected_items.begin(), expected_items.end());
  REQUIRE(expected_keys == std::vector<key_t>(output_keys_it));
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

struct DeviceMergeSort_SortPairsCopy_CustomType_Fixture_Tag;
C2H_TEST("DeviceMergeSort:SortPairsCopy works with custom types", "[merge_sort]")
{
  const size_t num_items = GENERATE_COPY(take(2, random(1, 100000)), values({5, 10000, 100000}));
  operation_t op         = make_operation(
    "op",
    "struct key_pair { short a; size_t b; };\n"
            "extern \"C\" __device__ void op(void* lhs_ptr, void* rhs_ptr, bool* out_ptr) {\n"
            "  key_pair* lhs = static_cast<key_pair*>(lhs_ptr);\n"
            "  key_pair* rhs = static_cast<key_pair*>(rhs_ptr);\n"
            "  bool* out = static_cast<bool*>(out_ptr);\n"
            "  *out = lhs->a == rhs->a ? lhs->b < rhs->b : lhs->a < rhs->a;\n"
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

  auto& build_cache    = get_cache<DeviceMergeSort_SortPairsCopy_CustomType_Fixture_Tag>();
  const auto& test_key = make_key<key_pair, item_pair>();

  merge_sort(input_keys_it, input_items_it, output_keys_it, output_items_it, num_items, op, build_cache, test_key);

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

struct DeviceMergeSort_SortPairsCopy_CustomType_WellKnown_Fixture_Tag;
C2H_TEST("DeviceMergeSort:SortPairsCopy works with custom types with well-known predicates", "[merge_sort][well_known]")
{
  const size_t num_items = GENERATE_COPY(take(2, random(1, 100000)), values({5, 10000, 100000}));
  operation_t op_state   = make_operation(
    "op",
    "struct key_pair { short a; size_t b; };\n"
      "extern \"C\" __device__ void op(void* lhs_ptr, void* rhs_ptr, bool* out_ptr) {\n"
      "  key_pair* lhs = static_cast<key_pair*>(lhs_ptr);\n"
      "  key_pair* rhs = static_cast<key_pair*>(rhs_ptr);\n"
      "  bool* out = static_cast<bool*>(out_ptr);\n"
      "  *out = lhs->a == rhs->a ? lhs->b < rhs->b : lhs->a < rhs->a;\n"
      "}");
  cccl_op_t op                = op_state;
  op.type                     = cccl_op_kind_t::CCCL_LESS;
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

  auto& build_cache    = get_cache<DeviceMergeSort_SortPairsCopy_CustomType_WellKnown_Fixture_Tag>();
  const auto& test_key = make_key<key_pair, item_pair>();

  merge_sort(input_keys_it, input_items_it, output_keys_it, output_items_it, num_items, op, build_cache, test_key);

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

struct DeviceMergeSort_SortKeys_Iterators_Fixture_Tag;
C2H_TEST("DeviceMergeSort::SortKeys works with input iterators", "[merge_sort]")
{
  using T             = int;
  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));

  operation_t op = make_operation("op", get_merge_sort_op(get_type_info<T>().type));
  iterator_t<T, random_access_iterator_state_t<T>> input_keys_it =
    make_random_access_iterator<T>(iterator_kind::INPUT, "int");
  std::vector<T> input_keys    = make_shuffled_sequence<T>(num_items);
  std::vector<T> expected_keys = input_keys;

  pointer_t<T> input_keys_ptr(input_keys);
  input_keys_it.state.data = input_keys_ptr.ptr;
  pointer_t<T> input_items_it;

  auto& build_cache    = get_cache<DeviceMergeSort_SortKeys_Iterators_Fixture_Tag>();
  const auto& test_key = make_key<T>();

  merge_sort(input_keys_it, input_items_it, input_keys_ptr, input_items_it, num_items, op, build_cache, test_key);

  std::sort(expected_keys.begin(), expected_keys.end());
  REQUIRE(expected_keys == std::vector<T>(input_keys_ptr));
}

struct DeviceMergeSort_SortPairs_Iterators_Fixture_Tag;
C2H_TEST("DeviceMergeSort::SortPairs works with input iterators", "[merge_sort]")
{
  using key_t         = int;
  using item_t        = int;
  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));

  operation_t op = make_operation("op", get_merge_sort_op(get_type_info<key_t>().type));
  iterator_t<key_t, random_access_iterator_state_t<key_t>> input_keys_it =
    make_random_access_iterator<key_t>(iterator_kind::INPUT, "int", "key");
  iterator_t<key_t, random_access_iterator_state_t<key_t>> input_items_it =
    make_random_access_iterator<key_t>(iterator_kind::INPUT, "int", "item");

  std::vector<key_t> input_keys = make_shuffled_sequence<key_t>(num_items);
  std::vector<item_t> input_items(num_items);
  std::transform(input_keys.begin(), input_keys.end(), input_items.begin(), [](key_t key) {
    return static_cast<item_t>(key);
  });

  std::vector<key_t> expected_keys   = input_keys;
  std::vector<item_t> expected_items = input_items;

  pointer_t<key_t> input_keys_ptr(input_keys);
  input_keys_it.state.data = input_keys_ptr.ptr;
  pointer_t<key_t> input_items_ptr(input_items);
  input_items_it.state.data = input_items_ptr.ptr;

  auto& build_cache    = get_cache<DeviceMergeSort_SortPairs_Iterators_Fixture_Tag>();
  const auto& test_key = make_key<key_t, item_t>();

  merge_sort(input_keys_it, input_items_it, input_keys_ptr, input_items_ptr, num_items, op, build_cache, test_key);

  std::sort(expected_keys.begin(), expected_keys.end());
  std::sort(expected_items.begin(), expected_items.end());
  REQUIRE(expected_keys == std::vector<key_t>(input_keys_ptr));
  REQUIRE(expected_items == std::vector<item_t>(input_items_ptr));
}

// These tests with output iterators are currently failing https://github.com/NVIDIA/cccl/issues/3722
#ifdef NEVER_DEFINED
C2H_TEST("DeviceMergeSort::SortKeys works with output iterators", "[merge_sort]")
{
  using TestType      = int;
  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));

  operation_t op = make_operation("op", get_merge_sort_op(get_type_info<TestType>().type));
  iterator_t<TestType, random_access_iterator_state_t> output_keys_it =
    make_iterator<TestType, random_access_iterator_state_t>(
      {"random_access_iterator_state_t", "struct random_access_iterator_state_t { int* d_input; };\n"},
      {"advance",
       "extern \"C\" __device__ void advance(random_access_iterator_state_t* state, unsigned long long offset) {\n"
       "  state->d_input += offset;\n"
       "}"},
      {"dereference",
       "extern \"C\" __device__ void dereference(random_access_iterator_state_t* state, int x) {\n"
       "  *state->d_input = x;\n"
       "}"});
  std::vector<TestType> input_keys    = make_shuffled_key_ranks_vector<TestType>(num_items);
  std::vector<TestType> expected_keys = input_keys;

  pointer_t<TestType> input_keys_it(input_keys);
  pointer_t<TestType> input_items_it;
  output_keys_it.state.d_input = input_keys_it.ptr;

  merge_sort(input_keys_it, input_items_it, output_keys_it, input_items_it, num_items, op);

  std::sort(expected_keys.begin(), expected_keys.end());
  REQUIRE(expected_keys == std::vector<TestType>(input_keys_it));
}

C2H_TEST("DeviceMergeSort::SortPairs works with output iterators for items", "[merge_sort]")
{
  using TestType      = int;
  using item_t        = int;
  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));

  operation_t op                   = make_operation("op", get_merge_sort_op(get_type_info<TestType>().type));
  std::vector<TestType> input_keys = make_shuffled_sequence<TestType>(num_items);
  std::vector<item_t> input_items(num_items);
  std::transform(input_keys.begin(), input_keys.end(), input_items.begin(), [](TestType key) {
    return static_cast<item_t>(key);
  });
  std::vector<TestType> expected_keys = input_keys;
  std::vector<item_t> expected_items  = input_items;

  iterator_t<item_t, item_random_access_iterator_state_t> output_items_it =
    make_iterator<TestType, item_random_access_iterator_state_t>(
      "struct item_random_access_iterator_state_t { int* d_input; };\n",
      {"advance",
       "extern \"C\" __device__ void advance(item_random_access_iterator_state_t* state, unsigned long long offset) "
       "{\n"
       "  state->d_input += offset;\n"
       "}"},
      {"dereference",
       "extern \"C\" __device__ void dereference(item_random_access_iterator_state_t* state, int x) {\n"
       "  *state->d_input = x;\n"
       "}"});

  pointer_t<TestType> input_keys_it(input_keys);
  pointer_t<item_t> input_items_it(input_items);
  output_items_it.state.d_input = input_items_it.ptr;

  merge_sort(input_keys_it, input_items_it, input_keys_it, output_items_it, num_items, op);

  std::sort(expected_keys.begin(), expected_keys.end());
  std::sort(expected_items.begin(), expected_items.end());
  REQUIRE(expected_keys == std::vector<TestType>(input_keys_it));
  REQUIRE(expected_items == std::vector<item_t>(input_items_it));
}

#endif

struct large_key_pair
{
  int a;
  char c[100];
};

C2H_TEST("MergeSort works with C++ source operations", "[merge_sort]")
{
  using key_t = int32_t;

  const std::size_t num_items = GENERATE(42, 1337, 42000);

  // Create operation from C++ source instead of LTO-IR
  std::string cpp_source = R"(
    extern "C" __device__ void op(void* lhs, void* rhs, void* result) {
      int* ilhs = (int*)lhs;
      int* irhs = (int*)rhs;
      bool* bresult = (bool*)result;
      *bresult = *ilhs < *irhs;
    }
  )";

  operation_t op = make_cpp_operation("op", cpp_source);

  std::vector<key_t> input_keys = make_shuffled_sequence<key_t>(num_items);
  pointer_t<key_t> input_keys_ptr(input_keys);
  pointer_t<key_t> output_keys_ptr(num_items);

  // Use int for items but won't actually use them
  pointer_t<int> input_items_ptr;
  pointer_t<int> output_items_ptr;

  // Test key including flag that this uses C++ source
  std::optional<std::string> test_key = std::format("cpp_source_test_{}_{}", num_items, typeid(key_t).name());

  auto& cache = fixture<merge_sort_build_cache_t, DeviceMergeSort_SortKeys_Fixture_Tag>::get_or_create().get_value();
  std::optional<merge_sort_build_cache_t> cache_opt = cache;

  merge_sort(input_keys_ptr, input_items_ptr, output_keys_ptr, output_items_ptr, num_items, op, cache_opt, test_key);

  const std::vector<key_t> output = output_keys_ptr;
  std::vector<key_t> expected     = input_keys;
  std::sort(expected.begin(), expected.end());
  REQUIRE(output == expected);
}

C2H_TEST("MergeSort works with C++ source operations using custom headers", "[merge_sort]")
{
  using key_t = int32_t;

  const std::size_t num_items = GENERATE(42, 1337, 42000);

  // Create operation from C++ source that uses the identity function from header
  std::string cpp_source = R"(
    #include "test_identity.h"
    extern "C" __device__ void op(void* lhs, void* rhs, void* result) {
      int* ilhs = (int*)lhs;
      int* irhs = (int*)rhs;
      bool* bresult = (bool*)result;
      int val_lhs = test_identity(*ilhs);
      int val_rhs = test_identity(*irhs);
      *bresult = val_lhs < val_rhs;
    }
  )";

  operation_t op = make_cpp_operation("op", cpp_source);

  std::vector<key_t> input_keys = make_shuffled_sequence<key_t>(num_items);
  pointer_t<key_t> input_keys_ptr(input_keys);
  pointer_t<key_t> output_keys_ptr(num_items);

  // Use int for items but won't actually use them
  pointer_t<int> input_items_ptr;
  pointer_t<int> output_items_ptr;

  // Test _ex version with custom build configuration
  cccl_build_config config;
  const char* extra_flags[]      = {"-DTEST_IDENTITY_ENABLED"};
  const char* extra_dirs[]       = {TEST_INCLUDE_PATH};
  config.extra_compile_flags     = extra_flags;
  config.num_extra_compile_flags = 1;
  config.extra_include_dirs      = extra_dirs;
  config.num_extra_include_dirs  = 1;

  // Build with _ex version
  cccl_device_merge_sort_build_result_t build;
  const auto& build_info = BuildInformation<>::init();
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_merge_sort_build_ex(
      &build,
      input_keys_ptr,
      input_items_ptr,
      output_keys_ptr,
      output_items_ptr,
      op,
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path(),
      &config));

  // Execute the merge sort
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_merge_sort(
      build,
      d_temp_storage,
      &temp_storage_bytes,
      input_keys_ptr,
      input_items_ptr,
      output_keys_ptr,
      output_items_ptr,
      num_items,
      op,
      CU_STREAM_LEGACY));
  pointer_t<char> temp_storage(temp_storage_bytes);
  d_temp_storage = static_cast<void*>(temp_storage.ptr);
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_merge_sort(
      build,
      d_temp_storage,
      &temp_storage_bytes,
      input_keys_ptr,
      input_items_ptr,
      output_keys_ptr,
      output_items_ptr,
      num_items,
      op,
      CU_STREAM_LEGACY));

  // Verify results
  std::vector<key_t> output_keys(num_items);
  cudaMemcpy(
    output_keys.data(), static_cast<void*>(output_keys_ptr.ptr), sizeof(key_t) * num_items, cudaMemcpyDeviceToHost);
  std::vector<key_t> expected_keys(num_items);
  cudaMemcpy(
    expected_keys.data(), static_cast<void*>(input_keys_ptr.ptr), sizeof(key_t) * num_items, cudaMemcpyDeviceToHost);
  std::sort(expected_keys.begin(), expected_keys.end());
  std::sort(expected_keys.begin(), expected_keys.end());
  REQUIRE(output_keys == expected_keys);

  // Cleanup
  REQUIRE(CUDA_SUCCESS == cccl_device_merge_sort_cleanup(&build));
}

// TODO: We no longer fail to build for large types due to no vsmem. Instead, the build passes,
// but we get a ptxas error about the kernel using too much shared memory.
/* C2H_TEST("DeviceMergeSort:SortPairsCopy fails to build for large types due to no vsmem", "[merge_sort]")
{
  const size_t num_items = 1;
  operation_t op         = make_operation(
    "op",
    "struct large_key_pair { int a; char c[100]; };\n"
            "extern \"C\" __device__ bool op(large_key_pair lhs, large_key_pair rhs) {\n"
            "  return lhs.a < rhs.a;\n"
            "}");
  const std::vector<int> a = generate<int>(num_items);
  std::vector<large_key_pair> input_keys(num_items);
  for (std::size_t i = 0; i < num_items; ++i)
  {
    input_keys[i] = large_key_pair{a[i], {}};
  }

  pointer_t<large_key_pair> input_keys_it(input_keys);
  pointer_t<int> input_items_it;

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
    CUDA_ERROR_UNKNOWN
    == cccl_device_merge_sort_build(
      &build,
      input_keys_it,
      input_items_it,
      input_keys_it,
      input_items_it,
      op,
      cc_major,
      cc_minor,
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path));
}
 */
