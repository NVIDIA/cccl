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
#include <optional> // std::optional
#include <string>

#include <cuda_runtime.h>

#include "algorithm_execution.h"
#include "build_result_caching.h"
#include "test_util.h"
#include <cccl/c/radix_sort.h>

using key_types =
  c2h::type_list<uint8_t,
                 int16_t,
                 uint32_t,
#if _CCCL_HAS_NVFP16()
                 __half,
#endif
                 double>;
using item_t = float;

template <typename KeyTy, typename ItemTy, bool descending = false, bool overwrite_okay = false>
struct TestParameters
{
  using KeyT                             = KeyTy;
  using ItemT                            = ItemTy;
  static constexpr bool m_descending     = descending;
  static constexpr bool m_overwrite_okay = overwrite_okay;

  constexpr TestParameters() = default;

  bool is_descending() const
  {
    return m_descending;
  }
  bool is_overwrite_okay() const
  {
    return m_overwrite_okay;
  }
};

using test_params_tuple =
  c2h::type_list<TestParameters<c2h::get<0, key_types>, item_t, false, false>,
                 TestParameters<c2h::get<1, key_types>, item_t, true, false>,
                 TestParameters<c2h::get<2, key_types>, item_t, false, true>,
                 TestParameters<c2h::get<3, key_types>, item_t, true, true>>;

using BuildResultT = cccl_device_radix_sort_build_result_t;

struct radix_sort_cleanup
{
  CUresult operator()(BuildResultT* build_data) const noexcept
  {
    return cccl_device_radix_sort_cleanup(build_data);
  }
};

using radix_sort_deleter       = BuildResultDeleter<BuildResultT, radix_sort_cleanup>;
using radix_sort_build_cache_t = build_cache_t<std::string, result_wrapper_t<BuildResultT, radix_sort_deleter>>;

template <typename Tag>
auto& get_cache()
{
  return fixture<radix_sort_build_cache_t, Tag>::get_or_create().get_value();
}

template <bool CheckSASS = true>
struct radix_sort_build
{
  static constexpr auto should_check_sass(int cc_major)
  {
    // TODO: re-enable w/ nvrtc version check
    return CheckSASS && cc_major < 9;
  }

  // operator arguments are (build_ptr, <all_args_of_algo_driver>, cc_major, cc_minor, <paths>)
  //   of all_args_of_algo_driver we pick out what gets passed to cccl_algo_build function
  CUresult operator()(
    BuildResultT* build_ptr,
    cccl_sort_order_t sort_order,
    cccl_iterator_t d_keys_in,
    cccl_iterator_t,
    cccl_iterator_t d_values_in,
    cccl_iterator_t,
    cccl_op_t decomposer,
    const char* decomposer_return_type,
    uint64_t,
    int,
    int,
    bool,
    int*,
    int cc_major,
    int cc_minor,
    const char* cub_path,
    const char* thrust_path,
    const char* libcudacxx_path,
    const char* ctk_path) const noexcept
  {
    return cccl_device_radix_sort_build(
      build_ptr,
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
      ctk_path);
  }
};

struct radix_sort_run
{
  template <typename... Rest>
  CUresult operator()(
    BuildResultT build,
    void* temp_storage,
    size_t* temp_storage_bytes,
    cccl_sort_order_t,
    cccl_iterator_t d_keys_in,
    cccl_iterator_t d_keys_out,
    cccl_iterator_t d_values_in,
    cccl_iterator_t d_values_out,
    cccl_op_t decomposer,
    const char*,
    Rest... rest) const noexcept
  {
    return cccl_device_radix_sort(
      build, temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, decomposer, rest...);
  }
};

template <bool CheckSASS = true, typename BuildCache = radix_sort_build_cache_t, typename KeyT = std::string>
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
  AlgorithmExecute<BuildResultT, radix_sort_build<CheckSASS>, radix_sort_cleanup, radix_sort_run, BuildCache, KeyT>(
    cache,
    lookup_key,
    sort_order,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    decomposer,
    decomposer_return_type,
    num_items,
    begin_bit,
    end_bit,
    is_overwrite_okay,
    selector);
}

struct DeviceRadixSort_SortKeys_Fixture_Tag;
C2H_TEST("DeviceRadixSort::SortKeys works", "[radix_sort]", test_params_tuple)
{
  using T     = c2h::get<0, TestType>;
  using KeyT  = typename T::KeyT;
  using ItemT = typename T::ItemT;

  constexpr auto this_test_params = T();
  // We want a mix of small and large sizes because different implementations will be called
  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));
  bool is_descending  = this_test_params.is_descending();
  const auto order    = is_descending ? CCCL_DESCENDING : CCCL_ASCENDING;

  const int begin_bit          = 0;
  const int end_bit            = sizeof(KeyT) * 8;
  const bool is_overwrite_okay = this_test_params.is_overwrite_okay();
  int selector                 = -1;

  static constexpr cccl_op_t decomposer_no_op{};
  static constexpr const char* unused_decomposer_retty = "";

  // problem descriptor: (order, TestType, item_t, is_overwrite_ok, items_present = false)
  std::vector<KeyT> input_keys    = make_shuffled_sequence<KeyT>(num_items);
  std::vector<KeyT> expected_keys = input_keys;

  pointer_t<KeyT> input_keys_it(input_keys);
  pointer_t<KeyT> output_keys_it(num_items);

  pointer_t<ItemT> input_items_it, output_items_it;

  auto& build_cache = get_cache<DeviceRadixSort_SortKeys_Fixture_Tag>();

  const std::string& key_string = KeyBuilder::join(
    {KeyBuilder::bool_as_key(is_descending),
     KeyBuilder::type_as_key<T>(),
     KeyBuilder::type_as_key<item_t>(),
     KeyBuilder::bool_as_key(is_overwrite_okay)});
  const auto& test_key = std::make_optional(key_string);

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
    std::sort(expected_keys.begin(), expected_keys.end(), std::greater<KeyT>());
  }
  else
  {
    std::sort(expected_keys.begin(), expected_keys.end());
  }

  auto& output_keys = (is_overwrite_okay && selector == 0) ? input_keys_it : output_keys_it;
  REQUIRE(expected_keys == std::vector<KeyT>(output_keys));
}

struct DeviceRadixSort_SortPairs_Fixture_Tag;
C2H_TEST("DeviceRadixSort::SortPairs works", "[radix_sort]", test_params_tuple)
{
  using T     = c2h::get<0, TestType>;
  using KeyT  = typename T::KeyT;
  using ItemT = typename T::ItemT;

  constexpr auto this_test_params = T();
  const int num_items             = GENERATE_COPY(take(2, random(1, 1000000)), values({500, 1000000, 2000000}));
  const bool is_descending        = this_test_params.is_descending();
  const auto order                = is_descending ? CCCL_DESCENDING : CCCL_ASCENDING;

  const int begin_bit          = 0;
  const int end_bit            = sizeof(KeyT) * 8;
  const bool is_overwrite_okay = this_test_params.is_overwrite_okay();
  int selector                 = -1;

  static constexpr cccl_op_t decomposer_no_op{};
  static constexpr const char* unused_decomposer_retty = "";

  // problem descriptor in this example: (order, TestType, item_t, is_overwrite_ok)

  std::vector<KeyT> input_keys = make_shuffled_sequence<KeyT>(num_items);
  std::vector<ItemT> input_items(num_items);
  std::transform(input_keys.begin(), input_keys.end(), input_items.begin(), [](KeyT key) {
    return static_cast<ItemT>(key);
  });

  std::vector<KeyT> expected_keys   = input_keys;
  std::vector<ItemT> expected_items = input_items;

  pointer_t<KeyT> input_keys_it(input_keys);
  pointer_t<KeyT> output_keys_it(num_items);

  pointer_t<ItemT> input_items_it(input_items);
  pointer_t<ItemT> output_items_it(num_items);

  auto& build_cache = get_cache<DeviceRadixSort_SortPairs_Fixture_Tag>();

  const std::string& key_string = KeyBuilder::join(
    {KeyBuilder::bool_as_key(is_descending),
     KeyBuilder::type_as_key<KeyT>(),
     KeyBuilder::type_as_key<ItemT>(),
     KeyBuilder::bool_as_key(is_overwrite_okay)});
  const auto& test_key = std::make_optional(key_string);

  radix_sort<false>(
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
    std::sort(expected_keys.begin(), expected_keys.end(), std::greater<KeyT>());
    std::sort(expected_items.begin(), expected_items.end(), std::greater<ItemT>());
  }
  else
  {
    std::sort(expected_keys.begin(), expected_keys.end());
    std::sort(expected_items.begin(), expected_items.end());
  }

  auto& output_keys  = (is_overwrite_okay && selector == 0) ? input_keys_it : output_keys_it;
  auto& output_items = (is_overwrite_okay && selector == 0) ? input_items_it : output_items_it;
  REQUIRE(expected_keys == std::vector<KeyT>(output_keys));
  REQUIRE(expected_items == std::vector<ItemT>(output_items));
}

C2H_TEST("RadixSort build result has AoT metadata populated", "[radix_sort][aot]")
{
  using T = int32_t;

  constexpr int device_id = 0;
  const auto& build_info  = BuildInformation<device_id>::init();

  pointer_t<T> keys_in(1);
  pointer_t<T> values_in(1);
  static constexpr cccl_op_t decomposer_no_op{};

  BuildResultT build{};
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_radix_sort_build(
      &build,
      CCCL_ASCENDING,
      keys_in,
      values_in,
      decomposer_no_op,
      "",
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path()));

  CHECK(build.cc == build_info.get_cc_major() * 10 + build_info.get_cc_minor());
  CHECK(build.cubin != nullptr);
  CHECK(build.cubin_size > 0);
  CHECK(build.runtime_policy != nullptr);
  CHECK(build.runtime_policy_size > 0);
  REQUIRE(build.single_tile_kernel_lowered_name != nullptr);
  CHECK(build.single_tile_kernel_lowered_name[0] != '\0');
  REQUIRE(build.upsweep_kernel_lowered_name != nullptr);
  CHECK(build.upsweep_kernel_lowered_name[0] != '\0');
  REQUIRE(build.downsweep_kernel_lowered_name != nullptr);
  CHECK(build.downsweep_kernel_lowered_name[0] != '\0');

  REQUIRE(CUDA_SUCCESS == cccl_device_radix_sort_cleanup(&build));
}

C2H_TEST("RadixSort compile/load round-trip", "[radix_sort][aot]")
{
  using T = int32_t;

  constexpr int device_id = 0;
  const auto& build_info  = BuildInformation<device_id>::init();

  pointer_t<T> dummy_keys_in(1);
  pointer_t<T> dummy_values_in(1);
  static constexpr cccl_op_t decomposer_no_op{};

  BuildResultT build{};
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_radix_sort_compile(
      &build,
      CCCL_ASCENDING,
      dummy_keys_in,
      dummy_values_in,
      decomposer_no_op,
      "",
      build_info.get_cc_major(),
      build_info.get_cc_minor(),
      build_info.get_cub_path(),
      build_info.get_thrust_path(),
      build_info.get_libcudacxx_path(),
      build_info.get_ctk_path(),
      nullptr));

  REQUIRE(build.cubin != nullptr);
  REQUIRE(build.cubin_size > 0);
  REQUIRE(build.single_tile_kernel_lowered_name != nullptr);
  REQUIRE(build.upsweep_kernel_lowered_name != nullptr);
  REQUIRE(build.downsweep_kernel_lowered_name != nullptr);
  CHECK(build.library == nullptr);
  CHECK(build.single_tile_kernel == nullptr);

  REQUIRE(CUDA_SUCCESS == cccl_device_radix_sort_load(&build));
  REQUIRE(build.library != nullptr);
  CHECK(build.single_tile_kernel != nullptr);
  CHECK(build.upsweep_kernel != nullptr);
  CHECK(build.downsweep_kernel != nullptr);

  constexpr std::size_t n    = 16;
  const std::vector<T> input = generate<T>(n);
  pointer_t<T> keys_in(input);
  pointer_t<T> keys_out(n);
  pointer_t<T> values_in(n);
  pointer_t<T> values_out(n);
  CUstream null_stream      = nullptr;
  size_t temp_storage_bytes = 0;
  int selector              = -1;

  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_radix_sort(
      build,
      nullptr,
      &temp_storage_bytes,
      keys_in,
      keys_out,
      values_in,
      values_out,
      decomposer_no_op,
      n,
      /*begin_bit=*/0,
      /*end_bit=*/sizeof(T) * 8,
      /*is_overwrite_okay=*/false,
      &selector,
      null_stream));
  pointer_t<uint8_t> temp_storage(temp_storage_bytes);
  REQUIRE(
    CUDA_SUCCESS
    == cccl_device_radix_sort(
      build,
      temp_storage.ptr,
      &temp_storage_bytes,
      keys_in,
      keys_out,
      values_in,
      values_out,
      decomposer_no_op,
      n,
      /*begin_bit=*/0,
      /*end_bit=*/sizeof(T) * 8,
      /*is_overwrite_okay=*/false,
      &selector,
      null_stream));

  std::vector<T> expected(input);
  std::sort(expected.begin(), expected.end());
  // is_overwrite_okay=false → result is always in keys_out (selector not used for routing)
  REQUIRE(expected == std::vector<T>(keys_out));

  REQUIRE(CUDA_SUCCESS == cccl_device_radix_sort_cleanup(&build));
}
