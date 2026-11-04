// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_select.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/__device/compute_capability.h>
#include <cuda/__execution/tune.h>
#include <cuda/iterator>
#include <cuda/std/execution>

#include <algorithm>
#include <sstream>

#include "block_size_extracting_helpers.h"
#include "catch2_test_custom_streams.cuh"
#include "catch2_test_device_select_common.cuh"
#include "catch2_test_launch_helper.h"

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSelect::UniqueByKey, device_select_unique_by_key);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include "cub_test_macros.h"

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

using block_size_check_t = block_size_extracting_op<cuda::std::equal_to<>>;

template <class T>
inline T to_bound(const unsigned long long bound)
{
  return static_cast<T>(bound);
}

template <>
inline ulonglong2 to_bound(const unsigned long long bound)
{
  return {bound, bound};
}

_CCCL_SUPPRESS_DEPRECATED_PUSH
_CCCL_SUPPRESS_DEPRECATED_NVRTC_DIAG
template <>
inline ulonglong4 to_bound(const unsigned long long bound)
{
  return {bound, bound, bound, bound};
}
_CCCL_SUPPRESS_DEPRECATED_POP

#  if _CCCL_CTK_AT_LEAST(13, 0)
template <>
inline ulonglong4_16a to_bound(const unsigned long long bound)
{
  return {bound, bound, bound, bound};
}
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

template <>
inline long2 to_bound(const unsigned long long bound)
{
  return {static_cast<long>(bound), static_cast<long>(bound)};
}

template <>
inline c2h::custom_type_t<c2h::equal_comparable_t> to_bound(const unsigned long long bound)
{
  c2h::custom_type_t<c2h::equal_comparable_t> val;
  val.key = bound;
  val.val = bound;
  return val;
}
template <typename EqualityOpT>
struct project_first
{
  EqualityOpT equality_op;
  template <typename Tuple>
  __host__ __device__ bool operator()(const Tuple& lhs, const Tuple& rhs) const
  {
    return equality_op(cuda::std::get<0>(lhs), cuda::std::get<0>(rhs));
  }
};

template <typename T>
struct custom_equality_op
{
  T div_val;
  __host__ __device__ __forceinline__ bool operator()(const T& lhs, const T& rhs) const
  {
    return (lhs / div_val) == (rhs / div_val);
  }
};

CUB_TEST("DeviceSelect::UniqueByKey works with user provided memory and environment",
         "[device][select_unique_by_key]",
         CUB_SMALL)
{
  using type     = int;
  using val_type = c2h::custom_type_t<c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> keys_in(num_items, thrust::default_init);
  c2h::device_vector<val_type> vals_in(num_items, thrust::default_init);
  c2h::device_vector<type> keys_out(num_items, thrust::default_init);
  c2h::device_vector<val_type> vals_out(num_items, thrust::default_init);
  c2h::gen(C2H_SEED(2), keys_in, to_bound<type>(0), to_bound<type>(42));
  c2h::gen(C2H_SEED(1), vals_in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  size_t expected_allocation_size = 0;
  auto error                      = cub::DeviceSelect::UniqueByKey(
    static_cast<void*>(nullptr),
    expected_allocation_size,
    keys_in.begin(),
    vals_in.begin(),
    keys_out.begin(),
    vals_out.begin(),
    d_first_num_selected_out,
    num_items);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  auto d_temp        = c2h::device_vector<uint8_t>(expected_allocation_size, thrust::no_init);
  void* temp_storage = thrust::raw_pointer_cast(d_temp.data());

  auto test_unique_by_key = [&](const auto& env) {
    size_t num_bytes = 0;
    error            = cub::DeviceSelect::UniqueByKey(
      static_cast<void*>(nullptr),
      num_bytes,
      keys_in.begin(),
      vals_in.begin(),
      keys_out.begin(),
      vals_out.begin(),
      d_first_num_selected_out,
      num_items,
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(expected_allocation_size == num_bytes);

    error = cub::DeviceSelect::UniqueByKey(
      temp_storage,
      num_bytes,
      keys_in.begin(),
      vals_in.begin(),
      keys_out.begin(),
      vals_out.begin(),
      d_first_num_selected_out,
      num_items,
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    // Ensure that we create the same output as std
    c2h::host_vector<type> reference_keys     = keys_in;
    c2h::host_vector<val_type> reference_vals = vals_in;
    const auto zip_begin                      = cuda::make_zip_iterator(reference_keys.begin(), reference_vals.begin());
    const auto zip_end                        = cuda::make_zip_iterator(reference_keys.end(), reference_vals.end());
    const auto boundary =
      std::unique(zip_begin, zip_end, project_first<cuda::std::equal_to<>>{cuda::std::equal_to<>{}});

    keys_out.resize(num_selected_out[0]);
    vals_out.resize(num_selected_out[0]);
    reference_keys.resize(num_selected_out[0]);
    reference_vals.resize(num_selected_out[0]);
    REQUIRE(static_cast<int>(boundary - zip_begin) == num_selected_out[0]);
    REQUIRE(reference_keys == keys_out);
    REQUIRE(reference_vals == vals_out);
  };

  test_with_custom_streams(test_unique_by_key);
}

CUB_TEST("DeviceSelect::UniqueByKey works with user provided operator, memory and environment",
         "[device][select_unique_by_key]",
         CUB_SMALL)
{
  using type        = int;
  using custom_op_t = custom_equality_op<type>;
  using val_type    = c2h::custom_type_t<c2h::equal_comparable_t>;

  const int num_items = GENERATE_COPY(take(2, random(1, 1000000)));
  c2h::device_vector<type> keys_in(num_items, thrust::default_init);
  c2h::device_vector<val_type> vals_in(num_items, thrust::default_init);
  c2h::device_vector<type> keys_out(num_items, thrust::default_init);
  c2h::device_vector<val_type> vals_out(num_items, thrust::default_init);
  c2h::gen(C2H_SEED(2), keys_in, to_bound<type>(0), to_bound<type>(42));
  c2h::gen(C2H_SEED(1), vals_in);

  // Needs to be device accessible
  c2h::device_vector<int> num_selected_out(1, 0);
  int* d_first_num_selected_out = thrust::raw_pointer_cast(num_selected_out.data());

  custom_op_t op{static_cast<type>(8)};

  size_t expected_allocation_size = 0;
  auto error                      = cub::DeviceSelect::UniqueByKey(
    static_cast<void*>(nullptr),
    expected_allocation_size,
    keys_in.begin(),
    vals_in.begin(),
    keys_out.begin(),
    vals_out.begin(),
    d_first_num_selected_out,
    num_items,
    op);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  auto d_temp        = c2h::device_vector<uint8_t>(expected_allocation_size, thrust::no_init);
  void* temp_storage = thrust::raw_pointer_cast(d_temp.data());

  auto test_unique_by_key = [&](const auto& env) {
    size_t num_bytes = 0;
    error            = cub::DeviceSelect::UniqueByKey(
      static_cast<void*>(nullptr),
      num_bytes,
      keys_in.begin(),
      vals_in.begin(),
      keys_out.begin(),
      vals_out.begin(),
      d_first_num_selected_out,
      num_items,
      op,
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(expected_allocation_size == num_bytes);

    error = cub::DeviceSelect::UniqueByKey(
      temp_storage,
      num_bytes,
      keys_in.begin(),
      vals_in.begin(),
      keys_out.begin(),
      vals_out.begin(),
      d_first_num_selected_out,
      num_items,
      op,
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    // Ensure that we create the same output as std
    c2h::host_vector<type> reference_keys     = keys_in;
    c2h::host_vector<val_type> reference_vals = vals_in;
    const auto zip_begin                      = cuda::make_zip_iterator(reference_keys.begin(), reference_vals.begin());
    const auto zip_end                        = cuda::make_zip_iterator(reference_keys.end(), reference_vals.end());
    const auto boundary                       = std::unique(zip_begin, zip_end, project_first<custom_op_t>{op});

    keys_out.resize(num_selected_out[0]);
    vals_out.resize(num_selected_out[0]);
    reference_keys.resize(num_selected_out[0]);
    reference_vals.resize(num_selected_out[0]);
    REQUIRE(static_cast<int>(boundary - zip_begin) == num_selected_out[0]);
    REQUIRE(reference_keys == keys_out);
    REQUIRE(reference_vals == vals_out);
  };

  test_with_custom_streams(test_unique_by_key);
}

CUB_TEST_CASE("Device select unique_by_key works with default environment", "[select][device]", CUB_SMALL)
{
  using value_t     = int;
  using num_items_t = int;

  const num_items_t num_items = 8;
  auto d_keys_in              = c2h::device_vector<value_t>{0, 2, 2, 9, 5, 5, 5, 8};
  auto d_values_in            = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_keys_out             = c2h::device_vector<value_t>(num_items);
  auto d_values_out           = c2h::device_vector<value_t>(num_items);
  auto d_num_selected         = c2h::device_vector<unsigned int>(1);

  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::UniqueByKey(
      d_keys_in.begin(),
      d_values_in.begin(),
      d_keys_out.begin(),
      d_values_out.begin(),
      d_num_selected.begin(),
      num_items));

  const c2h::device_vector<value_t> expected_keys{0, 2, 9, 5, 8};
  const c2h::device_vector<value_t> expected_values{1, 2, 4, 5, 8};
  const c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_keys_out.resize(d_num_selected[0]);
  d_values_out.resize(d_num_selected[0]);
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_values_out == expected_values);
}

CUB_TEST_CASE("Device select unique_by_key works with default environment and explicit env",
              "[select][device]",
              CUB_SMALL)
{
  using value_t     = int;
  using num_items_t = int;

  const num_items_t num_items = 8;
  auto d_keys_in              = c2h::device_vector<value_t>{0, 2, 2, 9, 5, 5, 5, 8};
  auto d_values_in            = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_keys_out             = c2h::device_vector<value_t>(num_items);
  auto d_values_out           = c2h::device_vector<value_t>(num_items);
  auto d_num_selected         = c2h::device_vector<int>(1);

  auto env = stdexec::env{};

  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::UniqueByKey(
      d_keys_in.begin(),
      d_values_in.begin(),
      d_keys_out.begin(),
      d_values_out.begin(),
      d_num_selected.begin(),
      num_items,
      env));

  const c2h::device_vector<value_t> expected_keys{0, 2, 9, 5, 8};
  const c2h::device_vector<value_t> expected_values{1, 2, 4, 5, 8};
  const c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_keys_out.resize(d_num_selected[0]);
  d_values_out.resize(d_num_selected[0]);
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_values_out == expected_values);
}

CUB_TEST_CASE("Device select unique_by_key default tuning chooses target block size", "[select][device]", CUB_SMALL)
{
  using num_items_t = int;
  using key_t       = int;
  using value_t     = int;

  using selector_t = cub::detail::unique_by_key::policy_selector_from_types<key_t, value_t>;

  cuda::compute_capability cc{};
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc));

  const auto target_block_size = selector_t{}(cc).threads_per_block;

  const num_items_t num_items = 1;
  auto d_keys_in              = c2h::device_vector<key_t>{0};
  auto d_keys_out             = c2h::device_vector<key_t>(1);
  auto d_values_out           = c2h::device_vector<value_t>(1);
  auto d_num_selected         = c2h::device_vector<unsigned int>(1);
  auto d_block_size           = c2h::device_vector<unsigned int>(1);
  const block_size_check_t equality_op{thrust::raw_pointer_cast(d_block_size.data())};
  auto d_values_in = cuda::constant_iterator(value_t{1});

  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::UniqueByKey(
      d_keys_in.begin(),
      d_values_in,
      d_keys_out.begin(),
      d_values_out.begin(),
      d_num_selected.begin(),
      num_items,
      equality_op));

  REQUIRE(d_num_selected[0] == 1);
  REQUIRE(d_keys_out[0] == key_t{0});
  REQUIRE(d_values_out[0] == value_t{1});
  REQUIRE(d_block_size[0] == static_cast<unsigned int>(target_block_size));
}

#endif // TEST_LAUNCH == 0

CUB_TEST("Device select unique_by_key uses environment", "[select][device]", CUB_SMALL)
{
  using value_t     = int;
  using num_items_t = int;

  const num_items_t num_items = 10;
  auto d_keys_in              = c2h::device_vector<value_t>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  auto d_values_in            = c2h::device_vector<value_t>{10, 11, 20, 21, 30, 31, 40, 41, 50, 51};
  auto d_keys_out             = c2h::device_vector<value_t>(num_items);
  auto d_values_out           = c2h::device_vector<value_t>(num_items);
  auto d_num_selected         = c2h::device_vector<unsigned int>(1);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::UniqueByKey(
      nullptr,
      expected_bytes_allocated,
      d_keys_in.begin(),
      d_values_in.begin(),
      d_keys_out.begin(),
      d_values_out.begin(),
      d_num_selected.begin(),
      num_items));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_select_unique_by_key(
    d_keys_in.begin(),
    d_values_in.begin(),
    d_keys_out.begin(),
    d_values_out.begin(),
    d_num_selected.begin(),
    num_items,
    ::cuda::std::equal_to<>{},
    env);

  const c2h::device_vector<value_t> expected_keys{1, 2, 3, 4, 5};
  const c2h::device_vector<value_t> expected_values{10, 20, 30, 40, 50};
  const c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_keys_out.resize(d_num_selected[0]);
  d_values_out.resize(d_num_selected[0]);
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_values_out == expected_values);
}

CUB_TEST("Device select unique_by_key uses environment without equality_op", "[select][device]", CUB_SMALL)
{
  using value_t     = int;
  using num_items_t = int;

  const num_items_t num_items = 10;
  auto d_keys_in              = c2h::device_vector<value_t>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  auto d_values_in            = c2h::device_vector<value_t>{10, 11, 20, 21, 30, 31, 40, 41, 50, 51};
  auto d_keys_out             = c2h::device_vector<value_t>(num_items);
  auto d_values_out           = c2h::device_vector<value_t>(num_items);
  auto d_num_selected         = c2h::device_vector<int>(1);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::UniqueByKey(
      nullptr,
      expected_bytes_allocated,
      d_keys_in.begin(),
      d_values_in.begin(),
      d_keys_out.begin(),
      d_values_out.begin(),
      d_num_selected.begin(),
      num_items));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_select_unique_by_key(
    d_keys_in.begin(),
    d_values_in.begin(),
    d_keys_out.begin(),
    d_values_out.begin(),
    d_num_selected.begin(),
    num_items,
    env);

  const c2h::device_vector<value_t> expected_keys{1, 2, 3, 4, 5};
  const c2h::device_vector<value_t> expected_values{10, 20, 30, 40, 50};
  const c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_keys_out.resize(d_num_selected[0]);
  d_values_out.resize(d_num_selected[0]);
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_values_out == expected_values);
}

CUB_TEST_CASE("Device select unique_by_key uses custom stream", "[select][device]", CUB_SMALL)
{
  using value_t     = int;
  using num_items_t = int;

  const num_items_t num_items = 8;
  auto d_keys_in              = c2h::device_vector<value_t>{0, 2, 2, 9, 5, 5, 5, 8};
  auto d_values_in            = c2h::device_vector<value_t>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_keys_out             = c2h::device_vector<value_t>(num_items);
  auto d_values_out           = c2h::device_vector<value_t>(num_items);
  auto d_num_selected         = c2h::device_vector<unsigned int>(1);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::UniqueByKey(
      nullptr,
      expected_bytes_allocated,
      d_keys_in.begin(),
      d_values_in.begin(),
      d_keys_out.begin(),
      d_values_out.begin(),
      d_num_selected.begin(),
      num_items));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  device_select_unique_by_key(
    d_keys_in.begin(),
    d_values_in.begin(),
    d_keys_out.begin(),
    d_values_out.begin(),
    d_num_selected.begin(),
    num_items,
    ::cuda::std::equal_to<>{},
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  const c2h::device_vector<value_t> expected_keys{0, 2, 9, 5, 8};
  const c2h::device_vector<value_t> expected_values{1, 2, 4, 5, 8};
  const c2h::device_vector<int> expected_num_selected{5};

  REQUIRE(d_num_selected == expected_num_selected);
  d_keys_out.resize(d_num_selected[0]);
  d_values_out.resize(d_num_selected[0]);
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_values_out == expected_values);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

#if TEST_LAUNCH != 1

template <unsigned int BlockThreads>
struct unique_by_key_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::UniqueByKeyPolicy
  {
    return {static_cast<int>(BlockThreads),
            10,
            cub::BLOCK_LOAD_DIRECT,
            cub::LOAD_DEFAULT,
            cub::BLOCK_SCAN_WARP_SCANS,
            cub::LookbackDelayPolicy{cub::LookbackDelayAlgorithm::fixed_delay, 350, 450}};
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

CUB_TEST("DeviceSelect::UniqueByKey can be tuned", "[select][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_keys_in                           = c2h::device_vector<int>{0, 0, 1, 1, 2, 2, 3, 3};
  auto d_values_in                         = c2h::device_vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto d_keys_out                          = c2h::device_vector<int>(8);
  auto d_values_out                        = c2h::device_vector<int>(8);
  auto d_num_selected                      = c2h::device_vector<unsigned int>(1);
  auto d_block_size                        = c2h::device_vector<unsigned int>(1);

  using eq_op_t = block_size_extracting_op<cuda::std::equal_to<>>;
  const eq_op_t equality_op{thrust::raw_pointer_cast(d_block_size.data())};

  auto env = cuda::execution::tune(unique_by_key_tuning<target_block_size>{});

  device_select_unique_by_key(
    d_keys_in.begin(),
    d_values_in.begin(),
    d_keys_out.begin(),
    d_values_out.begin(),
    d_num_selected.begin(),
    8,
    equality_op,
    env);
  REQUIRE(d_num_selected[0] == 4);
  REQUIRE(d_block_size[0] == target_block_size);
}

#  if _CCCL_COMPILER(GCC, >=, 8) // gcc 7 cannot preserve constexpr-ness from p1 to p2
CUB_TEST("Test UniqueByKeyPolicy properties", "[select_unique_by_key][device]", CUB_SMALL)
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::UniqueByKeyPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::UniqueByKeyPolicy>);

  // aggregate init
  constexpr auto p1 = cub::UniqueByKeyPolicy{
    256,
    11,
    cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
    cub::CacheLoadModifier::LOAD_DEFAULT,
    cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS,
    cub::LookbackDelayPolicy{cub::LookbackDelayAlgorithm::fixed_delay, 350, 450}};

#    if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto p2 = cub::UniqueByKeyPolicy{
    .threads_per_block = 256,
    .items_per_thread  = 11,
    .load_algorithm    = cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
    .load_modifier     = cub::CacheLoadModifier::LOAD_DEFAULT,
    .scan_algorithm    = cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS,
    .lookback_delay    = cub::LookbackDelayPolicy{
      .kind = cub::LookbackDelayAlgorithm::fixed_delay, .delay = 350, .l2_write_latency = 450}};
#    else // _CCCL_STD_VER >= 2020
  constexpr auto p2 = p1;
#    endif // _CCCL_STD_VER >= 2020

  // comparison
  STATIC_REQUIRE(p1 == p2);
  STATIC_REQUIRE_FALSE(p1 != p2);

  auto to_string = [](const auto& p) {
    std::ostringstream os;
    os << p;
    return os.str();
  };
  REQUIRE(to_string(p1)
          == "UniqueByKeyPolicy { .threads_per_block = 256, .items_per_thread = 11"
             ", .load_algorithm = BLOCK_LOAD_DIRECT, .load_modifier = LOAD_DEFAULT"
             ", .scan_algorithm = BLOCK_SCAN_WARP_SCANS"
             ", .lookback_delay = LookbackDelayPolicy { .kind = LookbackDelayAlgorithm::fixed_delay"
             ", .delay = 350, .l2_write_latency = 450 } }");
}
#  endif // _CCCL_COMPILER(GCC, >=, 8)

#endif // TEST_LAUNCH != 1
