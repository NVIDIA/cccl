//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <string>

#include <cuda_runtime.h>

#include "algorithm_execution.h"
#include "build_result_caching.h"
#include "test_util.h"
#include <cccl/c/binary_search.h>

using BuildResultT = cccl_device_binary_search_build_result_t;

struct binary_search_cleanup
{
  CUresult operator()(BuildResultT* build_data) const noexcept
  {
    return cccl_device_binary_search_cleanup(build_data);
  }
};

static std::string mode_as_key(cccl_binary_search_mode_t mode)
{
  switch (mode)
  {
    case cccl_binary_search_mode_t::CCCL_BINARY_SEARCH_LOWER_BOUND:
      return "LOWER";
    case cccl_binary_search_mode_t::CCCL_BINARY_SEARCH_UPPER_BOUND:
      return "UPPER";
  }

  throw std::runtime_error("Invalid binary search mode");
}

template <typename T>
std::optional<std::string> make_binary_search_key(bool inclusive, cccl_binary_search_mode_t mode)
{
  const std::string parts[] = {KeyBuilder::type_as_key<T>(), KeyBuilder::bool_as_key(inclusive), mode_as_key(mode)};
  return KeyBuilder::join(parts);
}

using binary_search_deleter       = BuildResultDeleter<BuildResultT, binary_search_cleanup>;
using binary_search_build_cache_t = build_cache_t<std::string, result_wrapper_t<BuildResultT, binary_search_deleter>>;

template <typename Tag>
auto& get_cache()
{
  return fixture<binary_search_build_cache_t, Tag>::get_or_create().get_value();
}

struct binary_search_build
{
  CUresult operator()(
    BuildResultT* build_ptr,
    cccl_binary_search_mode_t mode,
    cccl_iterator_t data,
    uint64_t,
    cccl_iterator_t values,
    uint64_t,
    cccl_iterator_t out,
    cccl_op_t op,
    int cc_major,
    int cc_minor,
    const char* cub_path,
    const char* thrust_path,
    const char* libcudacxx_path,
    const char* ctk_path) const noexcept
  {
    return cccl_device_binary_search_build(
      build_ptr, mode, data, values, out, op, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path);
  }

  static constexpr bool should_check_sass(int)
  {
    return false;
  }
};

struct binary_search_run
{
  template <typename... Ts>
  CUresult operator()(BuildResultT build, void*, std::size_t*, cccl_binary_search_mode_t, Ts... args) const noexcept
  {
    return cccl_device_binary_search(build, args...);
  }
};

template <cccl_binary_search_mode_t Mode>
struct binary_search_wrapper
{
  static const constexpr auto mode = Mode;

  template <typename BuildCache = binary_search_build_cache_t, typename KeyT = std::string>
  void operator()(
    cccl_iterator_t data,
    uint64_t num_items,
    cccl_iterator_t values,
    uint64_t num_values,
    cccl_iterator_t output,
    cccl_op_t op,
    std::optional<BuildCache>& cache,
    const std::optional<KeyT>& lookup_key) const
  {
    AlgorithmExecute<BuildResultT, binary_search_build, binary_search_cleanup, binary_search_run, BuildCache, KeyT>(
      cache, lookup_key, mode, data, num_items, values, num_values, output, op);
  }
};

using lower_bound = binary_search_wrapper<cccl_binary_search_mode_t::CCCL_BINARY_SEARCH_LOWER_BOUND>;
using upper_bound = binary_search_wrapper<cccl_binary_search_mode_t::CCCL_BINARY_SEARCH_UPPER_BOUND>;

// ==============
//   Test section
// ==============

using integral_types = c2h::type_list<int32_t, uint32_t, int64_t, uint64_t>;

struct std_lower_bound_t
{
  template <typename RangeIteratorT, typename T, typename CompareOpT>
  RangeIteratorT operator()(RangeIteratorT first, RangeIteratorT last, const T& value, CompareOpT comp) const
  {
    return std::lower_bound(first, last, value, comp);
  }
} std_lower_bound;

struct std_upper_bound_t
{
  template <typename RangeIteratorT, typename T, typename CompareOpT>
  RangeIteratorT operator()(RangeIteratorT first, RangeIteratorT last, const T& value, CompareOpT comp) const
  {
    return std::upper_bound(first, last, value, comp);
  }
} std_upper_bound;

template <typename Fixture, typename Value, typename Variant, typename HostVariant>
void test_vectorized(Variant variant, HostVariant host_variant)
{
  const std::size_t num_items = GENERATE(0, 43, take(4, random(1 << 12, 1 << 16)));
  operation_t op              = make_operation("op", get_merge_sort_op(get_type_info<Value>().type));

  const std::vector<Value> target_values = generate<Value>(num_items / 100);
  std::vector<Value> data                = generate<Value>(num_items);
  std::copy(target_values.begin(), target_values.end(), data.begin());
  std::sort(data.begin(), data.end());
  const std::vector<Value*> output(target_values.size(), nullptr);

  pointer_t<Value> target_values_ptr(target_values);
  pointer_t<Value> data_ptr(data);
  pointer_t<Value*> output_ptr(output);

  auto& build_cache    = get_cache<Fixture>();
  const auto& test_key = make_binary_search_key<Value>(true, Variant::mode);

  variant(data_ptr, num_items, target_values_ptr, target_values.size(), output_ptr, op, build_cache, test_key);

  std::vector<Value*> results(output_ptr);
  std::vector<Value*> expected(target_values.size(), nullptr);

  std::vector<std::ptrdiff_t> offsets(target_values.size(), 0);
  std::vector<std::ptrdiff_t> expected_offsets(target_values.size(), 0);

  for (auto i = 0u; i < target_values.size(); ++i)
  {
    offsets[i] = results[i] - data_ptr.ptr;
    expected_offsets[i] =
      host_variant(data.data(), data.data() + num_items, target_values[i], std::less<>()) - data.data();
  }

  CHECK(expected_offsets == offsets);
}

struct BinarySearch_IntegralTypes_LowerBound_Fixture_Tag;
C2H_TEST("DeviceFind::LowerBound works", "[find][device][binary-search]", integral_types)
{
  using value_type = c2h::get<0, TestType>;
  test_vectorized<BinarySearch_IntegralTypes_LowerBound_Fixture_Tag, value_type>(lower_bound{}, std_lower_bound);
}

struct BinarySearch_IntegralTypes_UpperBound_Fixture_Tag;
C2H_TEST("DeviceFind::UpperBound works", "[find][device][binary-search]", integral_types)
{
  using value_type = c2h::get<0, TestType>;
  test_vectorized<BinarySearch_IntegralTypes_UpperBound_Fixture_Tag, value_type>(upper_bound{}, std_upper_bound);
}
