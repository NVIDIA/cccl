//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <iostream> // std::cerr
#include <optional> // std::optional
#include <string>

#include <cuda_runtime.h>

#include "algorithm_execution.h"
#include "build_result_caching.h"
#include "test_util.h"
#include <cccl/c/scan.h>

using BuildResultT = cccl_device_scan_build_result_t;

struct scan_cleanup
{
  CUresult operator()(BuildResultT* build_data) const noexcept
  {
    return cccl_device_scan_cleanup(build_data);
  }
};

using scan_deleter       = BuildResultDeleter<BuildResultT, scan_cleanup>;
using scan_build_cache_t = build_cache_t<std::string, result_wrapper_t<BuildResultT, scan_deleter>>;

template <typename Tag>
auto& get_cache()
{
  return fixture<scan_build_cache_t, Tag>::get_or_create().get_value();
}

struct scan_build
{
  CUresult operator()(
    BuildResultT* build_ptr,
    bool inclusive,
    cccl_iterator_t input,
    cccl_iterator_t output,
    uint64_t,
    cccl_op_t op,
    cccl_value_t init,
    int cc_major,
    int cc_minor,
    const char* cub_path,
    const char* thrust_path,
    const char* libcudacxx_path,
    const char* ctk_path) const noexcept
  {
    return cccl_device_scan_build(
      build_ptr,
      input,
      output,
      op,
      init,
      inclusive,
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

struct scan_run
{
  template <typename... Ts>
  CUresult operator()(
    BuildResultT build, void* temp_storage, size_t* temp_storage_nbytes, bool inclusive, Ts... args) const noexcept
  {
    if (inclusive)
    {
      return cccl_device_inclusive_scan(build, temp_storage, temp_storage_nbytes, args...);
    }
    else
    {
      return cccl_device_exclusive_scan(build, temp_storage, temp_storage_nbytes, args...);
    }
  }
};

template <typename BuildCache = scan_build_cache_t, typename KeyT = std::string>
void scan(cccl_iterator_t input,
          cccl_iterator_t output,
          uint64_t num_items,
          cccl_op_t op,
          cccl_value_t init,
          bool inclusive,
          std::optional<BuildCache>& cache,
          const std::optional<KeyT>& lookup_key)
{
  AlgorithmExecute<BuildResultT, scan_build, scan_cleanup, scan_run, BuildCache, KeyT>(
    cache, lookup_key, inclusive, input, output, num_items, op, init);
}

// ==============
//   Test section
// ==============

using integral_types = c2h::type_list<int32_t, uint32_t, int64_t, uint64_t>;
struct Scan_IntegralTypes_Fixture_Tag;
C2H_TEST("Scan works with integral types", "[scan]", integral_types)
{
  using T = c2h::get<0, TestType>;

  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op              = make_operation("op", get_reduce_op(get_type_info<T>().type));
  const std::vector<T> input  = generate<T>(num_items);
  const std::vector<T> output(num_items, 0);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(output);
  value_t<T> init{T{42}};

  auto& build_cache    = get_cache<Scan_IntegralTypes_Fixture_Tag>();
  const auto& test_key = make_key<T>();

  scan(input_ptr, output_ptr, num_items, op, init, false, build_cache, test_key);

  std::vector<T> expected(num_items, 0);
  std::exclusive_scan(input.begin(), input.end(), expected.begin(), init.value);
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<T>(output_ptr));
  }
}

struct InclusiveScan_IntegralTypes_Fixture_Tag;
C2H_TEST("Inclusive Scan works with integral types", "[scan]", integral_types)
{
  using T = c2h::get<0, TestType>;

  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op              = make_operation("op", get_reduce_op(get_type_info<T>().type));
  const std::vector<T> input  = generate<T>(num_items);
  const std::vector<T> output(num_items, 0);
  pointer_t<T> input_ptr(input);
  pointer_t<T> output_ptr(output);
  value_t<T> init{T{42}};

  auto& build_cache    = get_cache<InclusiveScan_IntegralTypes_Fixture_Tag>();
  const auto& test_key = make_key<T>();

  scan(input_ptr, output_ptr, num_items, op, init, true, build_cache, test_key);

  std::vector<T> expected(num_items, 0);
  std::inclusive_scan(input.begin(), input.end(), expected.begin(), std::plus<>{}, init.value);
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<T>(output_ptr));
  }
}

struct pair
{
  short a;
  size_t b;

  bool operator==(const pair& other) const
  {
    return a == other.a && b == other.b;
  }
};

struct Scan_CustomTypes_Fixture_Tag;
C2H_TEST("Scan works with custom types", "[scan]")
{
  const std::size_t num_items = GENERATE(0, 42, take(4, random(1 << 12, 1 << 24)));

  operation_t op = make_operation(
    "op",
    "struct pair { short a; size_t b; };\n"
    "extern \"C\" __device__ void op(void* lhs_ptr, void* rhs_ptr, void* out_ptr) {\n"
    "  pair* lhs = static_cast<pair*>(lhs_ptr);\n"
    "  pair* rhs = static_cast<pair*>(rhs_ptr);\n"
    "  pair* out = static_cast<pair*>(out_ptr);\n"
    "  *out = pair{ lhs->a + rhs->a, lhs->b + rhs->b };\n"
    "}");
  const std::vector<short> a  = generate<short>(num_items);
  const std::vector<size_t> b = generate<size_t>(num_items);
  std::vector<pair> input(num_items);
  std::vector<pair> output(num_items);
  for (std::size_t i = 0; i < num_items; ++i)
  {
    input[i] = pair{a[i], b[i]};
  }
  pointer_t<pair> input_ptr(input);
  pointer_t<pair> output_ptr(output);
  value_t<pair> init{pair{4, 2}};

  auto& build_cache    = get_cache<Scan_CustomTypes_Fixture_Tag>();
  const auto& test_key = make_key<pair>();

  scan(input_ptr, output_ptr, num_items, op, init, false, build_cache, test_key);

  std::vector<pair> expected(num_items, {0, 0});
  std::exclusive_scan(input.begin(), input.end(), expected.begin(), init.value, [](const pair& lhs, const pair& rhs) {
    return pair{short(lhs.a + rhs.a), lhs.b + rhs.b};
  });
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<pair>(output_ptr));
  }
}

struct Scan_InputIterators_Fixture_Tag;
C2H_TEST("Scan works with input iterators", "[scan]")
{
  const std::size_t num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op              = make_operation("op", get_reduce_op(get_type_info<int>().type));
  iterator_t<int, counting_iterator_state_t<int>> input_it = make_counting_iterator<int>("int");
  input_it.state.value                                     = 0;
  pointer_t<int> output_it(num_items);
  value_t<int> init{42};

  auto& build_cache    = get_cache<Scan_InputIterators_Fixture_Tag>();
  const auto& test_key = make_key<int>();

  scan(input_it, output_it, num_items, op, init, false, build_cache, test_key);

  // vector storing a sequence of values 0, 1, 2, ..., num_items - 1
  std::vector<int> input(num_items);
  std::iota(input.begin(), input.end(), 0);

  std::vector<int> expected(num_items);
  std::exclusive_scan(input.begin(), input.end(), expected.begin(), init.value);
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<int>(output_it));
  }
}

struct Scan_OutputIterators_Fixture_Tag;
C2H_TEST("Scan works with output iterators", "[scan]")
{
  const int num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op      = make_operation("op", get_reduce_op(get_type_info<int>().type));
  iterator_t<int, random_access_iterator_state_t<int>> output_it =
    make_random_access_iterator<int>(iterator_kind::OUTPUT, "int", "out", " * 2");
  const std::vector<int> input = generate<int>(num_items);
  pointer_t<int> input_it(input);
  pointer_t<int> inner_output_it(num_items);
  output_it.state.data = inner_output_it.ptr;
  value_t<int> init{42};

  auto& build_cache    = get_cache<Scan_OutputIterators_Fixture_Tag>();
  const auto& test_key = make_key<int>();

  scan(input_it, output_it, num_items, op, init, false, build_cache, test_key);

  std::vector<int> expected(num_items);
  std::exclusive_scan(input.begin(), input.end(), expected.begin(), init.value);

  std::transform(expected.begin(), expected.end(), expected.begin(), [](int x) {
    return x * 2;
  });
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<int>(inner_output_it));
  }
}

struct Scan_ReverseInputIterators_Fixture_Tag;
C2H_TEST("Scan works with reverse input iterators", "[scan]")
{
  const std::size_t num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op              = make_operation("op", get_reduce_op(get_type_info<int>().type));
  iterator_t<int, random_access_iterator_state_t<int>> input_it =
    make_reverse_iterator<int>(iterator_kind::INPUT, "int");
  std::vector<int> input = generate<int>(num_items);
  pointer_t<int> input_ptr(input);
  input_it.state.data = input_ptr.ptr + num_items - 1;
  pointer_t<int> output_it(num_items);
  value_t<int> init{42};

  auto& build_cache    = get_cache<Scan_ReverseInputIterators_Fixture_Tag>();
  const auto& test_key = make_key<int>();

  scan(input_it, output_it, num_items, op, init, false, build_cache, test_key);

  std::vector<int> expected(num_items);
  std::exclusive_scan(input.rbegin(), input.rend(), expected.begin(), init.value);
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<int>(output_it));
  }
}

struct Scan_ReverseOutputIterators_Fixture_Tag;
C2H_TEST("Scan works with reverse output iterators", "[scan]")
{
  const int num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op      = make_operation("op", get_reduce_op(get_type_info<int>().type));
  iterator_t<int, random_access_iterator_state_t<int>> output_it =
    make_reverse_iterator<int>(iterator_kind::OUTPUT, "int", "out");
  const std::vector<int> input = generate<int>(num_items);
  pointer_t<int> input_it(input);
  pointer_t<int> inner_output_it(num_items);
  output_it.state.data = inner_output_it.ptr + num_items - 1;
  value_t<int> init{42};

  auto& build_cache    = get_cache<Scan_ReverseOutputIterators_Fixture_Tag>();
  const auto& test_key = make_key<int>();

  scan(input_it, output_it, num_items, op, init, false, build_cache, test_key);

  std::vector<int> expected(num_items);
  std::exclusive_scan(input.begin(), input.end(), expected.rbegin(), init.value);

  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<int>(inner_output_it));
  }
}

struct Scan_InputOutputIterators_Fixture_Tag;
C2H_TEST("Scan works with input and output iterators", "[scan]")
{
  const int num_items = GENERATE(1, 42, take(4, random(1 << 12, 1 << 16)));
  operation_t op      = make_operation("op", get_reduce_op(get_type_info<int>().type));
  iterator_t<int, constant_iterator_state_t<int>> input_it = make_constant_iterator<int>("int");
  input_it.state.value                                     = 1;
  iterator_t<int, random_access_iterator_state_t<int>> output_it =
    make_random_access_iterator<int>(iterator_kind::OUTPUT, "int", "out", " * 2");
  pointer_t<int> inner_output_it(num_items);
  output_it.state.data = inner_output_it.ptr;
  value_t<int> init{42};

  auto& build_cache    = get_cache<Scan_InputOutputIterators_Fixture_Tag>();
  const auto& test_key = make_key<int>();

  scan(input_it, output_it, num_items, op, init, false, build_cache, test_key);

  std::vector<int> expected(num_items, 1);
  std::exclusive_scan(expected.begin(), expected.end(), expected.begin(), init.value);
  std::transform(expected.begin(), expected.end(), expected.begin(), [](int x) {
    return x * 2;
  });
  if (num_items > 0)
  {
    REQUIRE(expected == std::vector<int>(inner_output_it));
  }
}
