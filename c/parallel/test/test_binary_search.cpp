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
  CUresult operator()(
    BuildResultT build, void* scratch, std::size_t* scratch_size, cccl_binary_search_mode_t, Ts... args) const noexcept
  {
    *scratch_size = 1;
    return (scratch) ? cccl_device_binary_search(build, args...) : CUDA_SUCCESS;
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
  const std::vector<std::ptrdiff_t> output(target_values.size(), 0);

  pointer_t<Value> target_values_ptr(target_values);
  pointer_t<Value> data_ptr(data);
  pointer_t<std::ptrdiff_t> output_ptr(output);

  auto& build_cache    = get_cache<Fixture>();
  const auto& test_key = make_binary_search_key<Value>(true, Variant::mode);

  variant(data_ptr, num_items, target_values_ptr, target_values.size(), output_ptr, op, build_cache, test_key);

  std::vector<std::ptrdiff_t> results(output_ptr);
  std::vector<std::ptrdiff_t> expected(target_values.size(), 0);

  std::vector<std::ptrdiff_t> expected_results(target_values.size(), 0);

  for (auto i = 0u; i < target_values.size(); ++i)
  {
    expected_results[i] =
      host_variant(data.data(), data.data() + num_items, target_values[i], std::less<>()) - data.data();
  }

  CHECK(expected_results == results);
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

#ifndef CCCL_C_PARALLEL_V2
C2H_TEST("BinarySearch build result has AoT metadata populated", "[binary_search][aot]")
{
  using T = int32_t;

  constexpr int device_id = 0;
  const auto& build_info  = BuildInformation<device_id>::init();

  cccl_op_t op = make_well_known_less_binary_predicate();
  pointer_t<T> data(1);
  pointer_t<T> values(1);
  pointer_t<T> out(1);

  BuildResultT build{};
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_binary_search_build(
      &build,
      CCCL_BINARY_SEARCH_LOWER_BOUND,
      data,
      values,
      out,
      op,
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path()));

  CHECK(build.transform.cc == build_info.get_cc_major() * 10 + build_info.get_cc_minor());
  CHECK((build.transform.payload != nullptr && build.transform.payload_kind == CCCL_PAYLOAD_CUBIN));
  CHECK(build.transform.payload_size > 0);
  REQUIRE(build.transform.transform_kernel_lowered_name != nullptr);
  CHECK(build.transform.transform_kernel_lowered_name[0] != '\0');

  REQUIRE(CUDA_SUCCESS == cccl_device_binary_search_cleanup(&build));
}

C2H_TEST("BinarySearch compile/load round-trip", "[binary_search][aot]")
{
  using T = int32_t;

  constexpr int device_id = 0;
  const auto& build_info  = BuildInformation<device_id>::init();

  operation_t op = make_operation("op", get_merge_sort_op(get_type_info<T>().type));
  pointer_t<T> dummy_data(1);
  pointer_t<T> dummy_values(1);
  pointer_t<std::ptrdiff_t> dummy_out(1);

  BuildResultT build{};
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_binary_search_compile(
      &build,
      CCCL_BINARY_SEARCH_LOWER_BOUND,
      dummy_data,
      dummy_values,
      dummy_out,
      op,
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path(),
      nullptr));

  REQUIRE((build.transform.payload != nullptr && build.transform.payload_kind == CCCL_PAYLOAD_CUBIN));
  REQUIRE(build.transform.payload_size > 0);
  REQUIRE(build.transform.transform_kernel_lowered_name != nullptr);
  CHECK(build.transform.library == nullptr);
  CHECK(build.transform.transform_kernel == nullptr);

  REQUIRE(CUDA_SUCCESS == cccl_device_binary_search_load(&build));
  REQUIRE(build.transform.library != nullptr);
  CHECK(build.transform.transform_kernel != nullptr);

  constexpr std::size_t n_items  = 16;
  constexpr std::size_t n_values = 4;
  std::vector<T> data            = generate<T>(n_items);
  std::sort(data.begin(), data.end());
  const std::vector<T> values = generate<T>(n_values);
  pointer_t<T> data_ptr(data);
  pointer_t<T> values_ptr(values);
  pointer_t<std::ptrdiff_t> output_ptr(n_values);
  CUstream null_stream = nullptr;

  REQUIRE(CUDA_SUCCESS
          == cccl_device_binary_search(build, data_ptr, n_items, values_ptr, n_values, output_ptr, op, null_stream));

  std::vector<std::ptrdiff_t> expected(n_values);
  for (std::size_t i = 0; i < n_values; ++i)
  {
    expected[i] = std::lower_bound(data.begin(), data.end(), values[i]) - data.begin();
  }
  REQUIRE(expected == std::vector<std::ptrdiff_t>(output_ptr));

  REQUIRE(CUDA_SUCCESS == cccl_device_binary_search_cleanup(&build));
}
#endif // CCCL_C_PARALLEL_V2
