// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_scan.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/__device/compute_capability.h>
#include <cuda/__execution/tune.h>
#include <cuda/__iterator/constant_iterator.h>
#include <cuda/memory_resource>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/utility>
#include <cuda/stream>

#include <sstream>

#include "block_size_extracting_helpers.h"
#include "catch2_test_custom_streams.cuh"
#include "catch2_test_launch_helper.h"
#include "catch2_test_memory_resources.h"
#include <c2h/device_and_stream.h>

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceScan::ExclusiveSumByKey, device_scan_exclusive_sum_by_key);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceScan::ExclusiveScanByKey, device_scan_exclusive_scan_by_key);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceScan::InclusiveSumByKey, device_scan_inclusive_sum_by_key);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceScan::InclusiveScanByKey, device_scan_inclusive_scan_by_key);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include "cub_test_macros.h"

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0
using block_size_check_t = block_size_extracting_op<cuda::std::plus<>>;

CUB_TEST_CASE("Device scan exclusive-sum-by-key works with default environment", "[scan][by_key][device]", CUB_SMALL)
{
  auto num_items = 7;
  auto d_keys    = thrust::device_vector<int>{0, 0, 1, 1, 1, 2, 2};
  auto d_in      = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_out     = thrust::device_vector<int>(num_items);

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSumByKey(d_keys.begin(), d_in.begin(), d_out.begin(), num_items));

  const thrust::device_vector<int> expected{0, 8, 0, 7, 12, 0, 0};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device scan exclusive-scan-by-key works with default environment", "[scan][by_key][device]", CUB_SMALL)
{
  using num_items_t = int;
  using key_t       = int;
  using value_t     = int;
  using accum_t     = value_t;

  using selector_t = cub::detail::scan_by_key::policy_selector_from_types<key_t, accum_t, value_t, block_size_check_t>;

  cudaDeviceProp device_props{};
  REQUIRE(cudaSuccess == cudaGetDeviceProperties(&device_props, c2h::current_device().get()));

  const auto target_block_size =
    selector_t{}(cuda::compute_capability{device_props.major, device_props.minor}).lookback.threads_per_block;

  const num_items_t num_items = 1;
  auto d_keys                 = thrust::device_vector<key_t>{0};
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};
  auto d_in  = cuda::constant_iterator(value_t{1});
  auto d_out = thrust::device_vector<value_t>(1);
  auto init  = value_t{0};

  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveScanByKey(d_keys.begin(), d_in, d_out.begin(), block_size_check, init, num_items));

  REQUIRE(d_out[0] == init);
  REQUIRE(d_block_size[0] == static_cast<unsigned int>(target_block_size));
}

CUB_TEST_CASE("Device scan inclusive-sum-by-key works with default environment", "[scan][by_key][device]", CUB_SMALL)
{
  auto num_items = 7;
  auto d_keys    = thrust::device_vector<int>{0, 0, 1, 1, 1, 2, 2};
  auto d_in      = thrust::device_vector<int>{8, 6, 7, 5, 3, 0, 9};
  auto d_out     = thrust::device_vector<int>(num_items);

  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveSumByKey(d_keys.begin(), d_in.begin(), d_out.begin(), num_items));

  const thrust::device_vector<int> expected{8, 14, 7, 12, 15, 0, 9};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device scan inclusive-scan-by-key works with default environment", "[scan][by_key][device]", CUB_SMALL)
{
  using num_items_t = int;
  using key_t       = int;
  using value_t     = int;
  using accum_t     = value_t;

  using selector_t = cub::detail::scan_by_key::policy_selector_from_types<key_t, accum_t, value_t, block_size_check_t>;

  cudaDeviceProp device_props{};
  REQUIRE(cudaSuccess == cudaGetDeviceProperties(&device_props, c2h::current_device().get()));

  const auto target_block_size =
    selector_t{}(cuda::compute_capability{device_props.major, device_props.minor}).lookback.threads_per_block;

  const num_items_t num_items = 1;
  auto d_keys                 = thrust::device_vector<key_t>{0};
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};
  auto d_in  = cuda::constant_iterator(value_t{1});
  auto d_out = thrust::device_vector<value_t>(1);

  REQUIRE(cudaSuccess
          == cub::DeviceScan::InclusiveScanByKey(d_keys.begin(), d_in, d_out.begin(), block_size_check, num_items));

  REQUIRE(d_out[0] == value_t{1});
  REQUIRE(d_block_size[0] == static_cast<unsigned int>(target_block_size));
}

#endif

#if TEST_LAUNCH != 1

template <int BlockThreads>
struct scan_by_key_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::ScanByKeyPolicy
  {
    return {cub::ScanByKeyAlgorithm::lookback,
            {BlockThreads,
             1,
             cub::BLOCK_LOAD_DIRECT,
             cub::LOAD_DEFAULT,
             cub::BLOCK_STORE_DIRECT,
             cub::BLOCK_SCAN_WARP_SCANS,
             {}}};
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;
using block_size_extracting_scan_op_t  = block_size_extracting_op<cuda::std::plus<>>;
using block_size_extracting_equality_t = block_size_extracting_op<cuda::std::equal_to<>>;

CUB_TEST("DeviceScan::ExclusiveSumByKey can be tuned", "[scan][by_key][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_keys{0, 0, 1, 1, 1, 2, 2};
  c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<int> d_out(7);
  c2h::device_vector<unsigned int> d_block_size(1);

  auto equality_op = block_size_extracting_equality_t{thrust::raw_pointer_cast(d_block_size.data())};
  auto env         = cuda::execution::tune(scan_by_key_tuning<target_block_size>{});

  device_scan_exclusive_sum_by_key(d_keys.begin(), d_in.begin(), d_out.begin(), 7, equality_op, env);

  const c2h::device_vector<int> expected{0, 8, 0, 7, 12, 0, 0};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceScan::ExclusiveScanByKey can be tuned", "[scan][by_key][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_keys{0, 0, 1, 1, 1, 2, 2};
  c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<int> d_out(7);
  c2h::device_vector<unsigned int> d_block_size(1);

  auto scan_op = block_size_extracting_scan_op_t{thrust::raw_pointer_cast(d_block_size.data())};
  auto env     = cuda::execution::tune(scan_by_key_tuning<target_block_size>{});

  device_scan_exclusive_scan_by_key(
    d_keys.begin(), d_in.begin(), d_out.begin(), scan_op, 0, 7, cuda::std::equal_to<>{}, env);

  const c2h::device_vector<int> expected{0, 8, 0, 7, 12, 0, 0};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceScan::InclusiveSumByKey can be tuned", "[scan][by_key][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_keys{0, 0, 1, 1, 1, 2, 2};
  c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<int> d_out(7);
  c2h::device_vector<unsigned int> d_block_size(1);

  auto equality_op = block_size_extracting_equality_t{thrust::raw_pointer_cast(d_block_size.data())};
  auto env         = cuda::execution::tune(scan_by_key_tuning<target_block_size>{});

  device_scan_inclusive_sum_by_key(d_keys.begin(), d_in.begin(), d_out.begin(), 7, equality_op, env);

  const c2h::device_vector<int> expected{8, 14, 7, 12, 15, 0, 9};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceScan::InclusiveScanByKey can be tuned", "[scan][by_key][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_keys{0, 0, 1, 1, 1, 2, 2};
  c2h::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<int> d_out(7);
  c2h::device_vector<unsigned int> d_block_size(1);

  auto scan_op = block_size_extracting_scan_op_t{thrust::raw_pointer_cast(d_block_size.data())};
  auto env     = cuda::execution::tune(scan_by_key_tuning<target_block_size>{});

  device_scan_inclusive_scan_by_key(
    d_keys.begin(), d_in.begin(), d_out.begin(), scan_op, 7, cuda::std::equal_to<>{}, env);

  const c2h::device_vector<int> expected{8, 14, 7, 12, 15, 0, 9};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH != 1

CUB_TEST("Device scan exclusive-sum-by-key uses environment", "[scan][by_key][device]", CUB_SMALL)
{
  using num_items_t = int;

  const num_items_t num_items = 7;
  auto d_keys                 = thrust::device_vector<int>{0, 0, 1, 1, 1, 2, 2};
  auto d_in                   = thrust::device_vector<float>{8.0f, 6.0f, 7.0f, 5.0f, 3.0f, 0.0f, 9.0f};
  auto d_out                  = thrust::device_vector<float>(num_items);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSumByKey(
      nullptr,
      expected_bytes_allocated,
      d_keys.begin(),
      d_in.begin(),
      d_out.begin(),
      num_items,
      cuda::std::equal_to<>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_scan_exclusive_sum_by_key(d_keys.begin(), d_in.begin(), d_out.begin(), num_items, cuda::std::equal_to<>{}, env);

  const thrust::device_vector<float> expected{0.0f, 8.0f, 0.0f, 7.0f, 12.0f, 0.0f, 0.0f};
  REQUIRE(d_out == expected);
}

CUB_TEST("Device scan exclusive-scan-by-key uses environment", "[scan][by_key][device]", CUB_SMALL)
{
  using scan_op_t   = cuda::std::plus<>;
  using num_items_t = int;

  const num_items_t num_items = 7;
  auto d_keys                 = thrust::device_vector<int>{0, 0, 1, 1, 1, 2, 2};
  auto d_in                   = thrust::device_vector<float>{8.0f, 6.0f, 7.0f, 5.0f, 3.0f, 0.0f, 9.0f};
  auto d_out                  = thrust::device_vector<float>(num_items);
  auto init                   = 0.0f;

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveScanByKey(
      nullptr,
      expected_bytes_allocated,
      d_keys.begin(),
      d_in.begin(),
      d_out.begin(),
      scan_op_t{},
      init,
      num_items,
      cuda::std::equal_to<>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_scan_exclusive_scan_by_key(
    d_keys.begin(), d_in.begin(), d_out.begin(), scan_op_t{}, init, num_items, cuda::std::equal_to<>{}, env);

  const thrust::device_vector<float> expected{0.0f, 8.0f, 0.0f, 7.0f, 12.0f, 0.0f, 0.0f};
  REQUIRE(d_out == expected);
}

CUB_TEST("Device scan inclusive-sum-by-key uses environment", "[scan][by_key][device]", CUB_SMALL)
{
  using num_items_t = int;

  const num_items_t num_items = 7;
  auto d_keys                 = thrust::device_vector<int>{0, 0, 1, 1, 1, 2, 2};
  auto d_in                   = thrust::device_vector<float>{8.0f, 6.0f, 7.0f, 5.0f, 3.0f, 0.0f, 9.0f};
  auto d_out                  = thrust::device_vector<float>(num_items);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveSumByKey(
      nullptr,
      expected_bytes_allocated,
      d_keys.begin(),
      d_in.begin(),
      d_out.begin(),
      num_items,
      cuda::std::equal_to<>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_scan_inclusive_sum_by_key(d_keys.begin(), d_in.begin(), d_out.begin(), num_items, cuda::std::equal_to<>{}, env);

  const thrust::device_vector<float> expected{8.0f, 14.0f, 7.0f, 12.0f, 15.0f, 0.0f, 9.0f};
  REQUIRE(d_out == expected);
}

CUB_TEST("Device scan inclusive-scan-by-key uses environment", "[scan][by_key][device]", CUB_SMALL)
{
  using scan_op_t   = cuda::std::plus<>;
  using num_items_t = int;

  const num_items_t num_items = 7;
  auto d_keys                 = thrust::device_vector<int>{0, 0, 1, 1, 1, 2, 2};
  auto d_in                   = thrust::device_vector<float>{8.0f, 6.0f, 7.0f, 5.0f, 3.0f, 0.0f, 9.0f};
  auto d_out                  = thrust::device_vector<float>(num_items);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveScanByKey(
      nullptr,
      expected_bytes_allocated,
      d_keys.begin(),
      d_in.begin(),
      d_out.begin(),
      scan_op_t{},
      num_items,
      cuda::std::equal_to<>{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_scan_inclusive_scan_by_key(
    d_keys.begin(), d_in.begin(), d_out.begin(), scan_op_t{}, num_items, cuda::std::equal_to<>{}, env);

  const thrust::device_vector<float> expected{8.0f, 14.0f, 7.0f, 12.0f, 15.0f, 0.0f, 9.0f};
  REQUIRE(d_out == expected);
}

#if TEST_LAUNCH == 0

// Explicit-storage calls bypass the launch wrappers, so test them only in the host-launch variant.
template <class TwoPhaseFn, class EnvT>
size_t run_two_phase(TwoPhaseFn two_phase, EnvT&& env, cudaStream_t stream = nullptr)
{
  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == two_phase(nullptr, temp_storage_bytes, cuda::std::forward<EnvT>(env)));

  c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  {
    const stream_scope scope{stream};
    REQUIRE(
      cudaSuccess
      == two_phase(thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, cuda::std::forward<EnvT>(env)));
  }
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));
  return temp_storage_bytes;
}

struct mutable_stream_convertible
{
  cudaStream_t stream;

  operator cudaStream_t() & noexcept
  {
    return stream;
  }
};

struct rvalue_stream_convertible
{
  cudaStream_t stream;

  operator cudaStream_t() && noexcept
  {
    return stream;
  }
};

template <class TwoPhaseFn>
void test_two_phase_env_kinds(size_t expected_temp_storage_bytes, TwoPhaseFn two_phase)
{
  const cuda::stream stream = c2h::make_current_device_stream();
  test_with_custom_streams(
    [&](const auto& env) {
      REQUIRE(run_two_phase(two_phase, env, stream.get()) == expected_temp_storage_bytes);
    },
    stream);

  SECTION("mutable stream conversion")
  {
    mutable_stream_convertible stream_arg{stream.get()};
    REQUIRE(run_two_phase(two_phase, stream_arg, stream.get()) == expected_temp_storage_bytes);
  }

  SECTION("rvalue stream conversion")
  {
    REQUIRE(
      run_two_phase(two_phase, rvalue_stream_convertible{stream.get()}, stream.get()) == expected_temp_storage_bytes);
  }

  SECTION("default environment")
  {
    REQUIRE(run_two_phase(two_phase, stdexec::env<>{}) == expected_temp_storage_bytes);
  }

  SECTION("memory resource is ignored")
  {
    const auto mr_env = stdexec::prop{cuda::mr::get_memory_resource_t{}, throwing_memory_resource{}};
    REQUIRE(run_two_phase(two_phase, stdexec::env{mr_env}) == expected_temp_storage_bytes);
  }

  SECTION("legacy nullptr stream")
  {
    REQUIRE(run_two_phase(two_phase, nullptr) == expected_temp_storage_bytes);
  }

  SECTION("legacy literal 0 stream")
  {
    REQUIRE(run_two_phase(two_phase, 0) == expected_temp_storage_bytes);
  }
}

struct key_group_equal
{
  __host__ __device__ bool operator()(cuda::std::int32_t lhs, cuda::std::int32_t rhs) const
  {
    return lhs / 2 == rhs / 2;
  }
};

CUB_TEST_CASE("DeviceScan::ExclusiveSumByKey works with user provided memory and environment",
              "[scan][by_key][device]",
              CUB_SMALL)
{
  constexpr cuda::std::int32_t num_items = 7;
  const c2h::device_vector<cuda::std::int32_t> d_keys{0, 1, 2, 3, 2, 4, 5};
  const c2h::device_vector<cuda::std::int32_t> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<cuda::std::int32_t> d_out(num_items, thrust::no_init);

  size_t expected_bytes{};
  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveSumByKey(
            nullptr, expected_bytes, d_keys.begin(), d_in.begin(), d_out.begin(), num_items));

  test_two_phase_env_kinds(expected_bytes, [&](void* storage, size_t& bytes, auto&& env) {
    return cub::DeviceScan::ExclusiveSumByKey(
      storage,
      bytes,
      d_keys.begin(),
      d_in.begin(),
      d_out.begin(),
      num_items,
      key_group_equal{},
      cuda::std::forward<decltype(env)>(env));
  });

  const c2h::device_vector<cuda::std::int32_t> expected{0, 8, 0, 7, 12, 0, 0};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("DeviceScan::ExclusiveScanByKey works with user provided memory and environment",
              "[scan][by_key][device]",
              CUB_SMALL)
{
  constexpr cuda::std::int32_t num_items = 7;
  const c2h::device_vector<cuda::std::int32_t> d_keys{0, 1, 2, 3, 2, 4, 5};
  const c2h::device_vector<cuda::std::int32_t> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<cuda::std::int32_t> d_out(num_items, thrust::no_init);

  size_t expected_bytes{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveScanByKey(
      nullptr, expected_bytes, d_keys.begin(), d_in.begin(), d_out.begin(), cuda::std::multiplies<>{}, 2, num_items));

  test_two_phase_env_kinds(expected_bytes, [&](void* storage, size_t& bytes, auto&& env) {
    return cub::DeviceScan::ExclusiveScanByKey(
      storage,
      bytes,
      d_keys.begin(),
      d_in.begin(),
      d_out.begin(),
      cuda::std::multiplies<>{},
      2,
      num_items,
      key_group_equal{},
      cuda::std::forward<decltype(env)>(env));
  });

  const c2h::device_vector<cuda::std::int32_t> expected{2, 16, 2, 14, 70, 2, 0};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("DeviceScan::InclusiveSumByKey works with user provided memory and environment",
              "[scan][by_key][device]",
              CUB_SMALL)
{
  constexpr cuda::std::int32_t num_items = 7;
  const c2h::device_vector<cuda::std::int32_t> d_keys{0, 1, 2, 3, 2, 4, 5};
  const c2h::device_vector<cuda::std::int32_t> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<cuda::std::int32_t> d_out(num_items, thrust::no_init);

  size_t expected_bytes{};
  REQUIRE(cudaSuccess
          == cub::DeviceScan::InclusiveSumByKey(
            nullptr, expected_bytes, d_keys.begin(), d_in.begin(), d_out.begin(), num_items));

  test_two_phase_env_kinds(expected_bytes, [&](void* storage, size_t& bytes, auto&& env) {
    return cub::DeviceScan::InclusiveSumByKey(
      storage,
      bytes,
      d_keys.begin(),
      d_in.begin(),
      d_out.begin(),
      num_items,
      key_group_equal{},
      cuda::std::forward<decltype(env)>(env));
  });

  const c2h::device_vector<cuda::std::int32_t> expected{8, 14, 7, 12, 15, 0, 9};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("DeviceScan::InclusiveScanByKey works with user provided memory and environment",
              "[scan][by_key][device]",
              CUB_SMALL)
{
  constexpr cuda::std::int32_t num_items = 7;
  const c2h::device_vector<cuda::std::int32_t> d_keys{0, 1, 2, 3, 2, 4, 5};
  const c2h::device_vector<cuda::std::int32_t> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<cuda::std::int32_t> d_out(num_items, thrust::no_init);

  size_t expected_bytes{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveScanByKey(
      nullptr, expected_bytes, d_keys.begin(), d_in.begin(), d_out.begin(), cuda::std::multiplies<>{}, num_items));

  test_two_phase_env_kinds(expected_bytes, [&](void* storage, size_t& bytes, auto&& env) {
    return cub::DeviceScan::InclusiveScanByKey(
      storage,
      bytes,
      d_keys.begin(),
      d_in.begin(),
      d_out.begin(),
      cuda::std::multiplies<>{},
      num_items,
      key_group_equal{},
      cuda::std::forward<decltype(env)>(env));
  });

  const c2h::device_vector<cuda::std::int32_t> expected{8, 48, 7, 35, 105, 0, 0};
  REQUIRE(d_out == expected);
}

CUB_TEST("DeviceScan::ExclusiveSumByKey can be tuned with user provided memory",
         "[scan][by_key][device]",
         CUB_SMALL,
         block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  const c2h::device_vector<cuda::std::int32_t> d_keys{0, 0, 1, 1, 1, 2, 2};
  const c2h::device_vector<cuda::std::int32_t> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<cuda::std::int32_t> d_out(7, thrust::no_init);
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_equality_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};
  const cuda::stream stream = c2h::make_current_device_stream();
  const auto env =
    stdexec::env{cuda::execution::tune(scan_by_key_tuning<target_block_size>{}), cuda::stream_ref{stream}};

  run_two_phase(
    [&](void* storage, size_t& bytes, const auto& env) {
      return cub::DeviceScan::ExclusiveSumByKey(
        storage, bytes, d_keys.begin(), d_in.begin(), d_out.begin(), 7, block_size_check, env);
    },
    env,
    stream.get());

  const c2h::device_vector<cuda::std::int32_t> expected{0, 8, 0, 7, 12, 0, 0};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceScan::ExclusiveScanByKey can be tuned with user provided memory",
         "[scan][by_key][device]",
         CUB_SMALL,
         block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  const c2h::device_vector<cuda::std::int32_t> d_keys{0, 0, 1, 1, 1, 2, 2};
  const c2h::device_vector<cuda::std::int32_t> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<cuda::std::int32_t> d_out(7, thrust::no_init);
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_scan_op_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};
  const cuda::stream stream = c2h::make_current_device_stream();
  const auto env =
    stdexec::env{cuda::execution::tune(scan_by_key_tuning<target_block_size>{}), cuda::stream_ref{stream}};

  run_two_phase(
    [&](void* storage, size_t& bytes, const auto& env) {
      return cub::DeviceScan::ExclusiveScanByKey(
        storage,
        bytes,
        d_keys.begin(),
        d_in.begin(),
        d_out.begin(),
        block_size_check,
        0,
        7,
        cuda::std::equal_to<>{},
        env);
    },
    env,
    stream.get());

  const c2h::device_vector<cuda::std::int32_t> expected{0, 8, 0, 7, 12, 0, 0};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceScan::InclusiveSumByKey can be tuned with user provided memory",
         "[scan][by_key][device]",
         CUB_SMALL,
         block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  const c2h::device_vector<cuda::std::int32_t> d_keys{0, 0, 1, 1, 1, 2, 2};
  const c2h::device_vector<cuda::std::int32_t> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<cuda::std::int32_t> d_out(7, thrust::no_init);
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_equality_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};
  const cuda::stream stream = c2h::make_current_device_stream();
  const auto env =
    stdexec::env{cuda::execution::tune(scan_by_key_tuning<target_block_size>{}), cuda::stream_ref{stream}};

  run_two_phase(
    [&](void* storage, size_t& bytes, const auto& env) {
      return cub::DeviceScan::InclusiveSumByKey(
        storage, bytes, d_keys.begin(), d_in.begin(), d_out.begin(), 7, block_size_check, env);
    },
    env,
    stream.get());

  const c2h::device_vector<cuda::std::int32_t> expected{8, 14, 7, 12, 15, 0, 9};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceScan::InclusiveScanByKey can be tuned with user provided memory",
         "[scan][by_key][device]",
         CUB_SMALL,
         block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  const c2h::device_vector<cuda::std::int32_t> d_keys{0, 0, 1, 1, 1, 2, 2};
  const c2h::device_vector<cuda::std::int32_t> d_in{8, 6, 7, 5, 3, 0, 9};
  c2h::device_vector<cuda::std::int32_t> d_out(7, thrust::no_init);
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_scan_op_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};
  const cuda::stream stream = c2h::make_current_device_stream();
  const auto env =
    stdexec::env{cuda::execution::tune(scan_by_key_tuning<target_block_size>{}), cuda::stream_ref{stream}};

  run_two_phase(
    [&](void* storage, size_t& bytes, const auto& env) {
      return cub::DeviceScan::InclusiveScanByKey(
        storage, bytes, d_keys.begin(), d_in.begin(), d_out.begin(), block_size_check, 7, cuda::std::equal_to<>{}, env);
    },
    env,
    stream.get());

  const c2h::device_vector<cuda::std::int32_t> expected{8, 14, 7, 12, 15, 0, 9};
  REQUIRE(d_out == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH == 0

#if _CCCL_COMPILER(GCC, >=, 8) // gcc 7 cannot preserve constexpr-ness from p1 to p2
CUB_TEST("Test ScanByKeyPolicy properties", "[scan][by_key][device]", CUB_SMALL)
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::ScanByKeyPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::ScanByKeyPolicy>);

  // aggregate init
  constexpr auto p1 = cub::ScanByKeyPolicy{
    cub::ScanByKeyAlgorithm::lookback,
    {256,
     11,
     cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
     cub::CacheLoadModifier::LOAD_DEFAULT,
     cub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT,
     cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING,
     cub::LookbackDelayPolicy{cub::LookbackDelayAlgorithm::fixed_delay, 832, 1165}}};

#  if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto p2 = cub::ScanByKeyPolicy{
    .algorithm = cub::ScanByKeyAlgorithm::lookback,
    .lookback  = cub::ScanByKeyLookbackPolicy{
      .threads_per_block = 256,
      .items_per_thread  = 11,
      .load_algorithm    = cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
      .load_modifier     = cub::CacheLoadModifier::LOAD_DEFAULT,
      .store_algorithm   = cub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT,
      .scan_algorithm    = cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING,
      .lookback_delay    = cub::LookbackDelayPolicy{
        .kind = cub::LookbackDelayAlgorithm::fixed_delay, .delay = 832, .l2_write_latency = 1165}}};
#  else // _CCCL_STD_VER >= 2020
  constexpr auto p2 = p1;
#  endif // _CCCL_STD_VER >= 2020

  // comparison
  STATIC_REQUIRE(p1 == p2);
  STATIC_REQUIRE_FALSE(p1 != p2);

  auto to_string = [](const auto& p) {
    std::ostringstream os;
    os << p;
    return os.str();
  };
  REQUIRE(to_string(p1)
          == "ScanByKeyPolicy { .algorithm = ScanByKeyAlgorithm::lookback"
             ", .lookback = ScanByKeyLookbackPolicy { .threads_per_block = 256, .items_per_thread = 11"
             ", .load_algorithm = BLOCK_LOAD_DIRECT, .load_modifier = LOAD_DEFAULT"
             ", .store_algorithm = BLOCK_STORE_DIRECT, .scan_algorithm = BLOCK_SCAN_RAKING"
             ", .lookback_delay = LookbackDelayPolicy { .kind = LookbackDelayAlgorithm::fixed_delay"
             ", .delay = 832, .l2_write_latency = 1165 } } }");
}
#endif // _CCCL_COMPILER(GCC, >=, 8)
