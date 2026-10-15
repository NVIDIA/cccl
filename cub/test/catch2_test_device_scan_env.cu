// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/__device/compute_capability.h>
#include <cuda/iterator>
#include <cuda/std/cstdint>

#include <sstream>

#include "block_size_extracting_helpers.h"
#include "catch2_test_custom_streams.cuh"
#include "catch2_test_device_scan.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/generators.h>

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceScan::ExclusiveScan, device_scan_exclusive);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceScan::ExclusiveSum, device_scan_exclusive_sum);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceScan::InclusiveScan, device_scan_inclusive);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceScan::InclusiveSum, device_scan_inclusive_sum);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceScan::InclusiveScanInit, device_scan_inclusive_init);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

#include "cub_test_macros.h"

namespace stdexec = cuda::std::execution;

using block_size_check_t = block_size_extracting_op<cuda::std::plus<>>;

// Launcher helper always passes an environment.
// We need a test of simple use to check if default environment works.
// ifdef it out not to spend time compiling and running it twice.
#if TEST_LAUNCH == 0

CUB_TEST_CASE("Device scan exclusive scan works with default environment", "[scan][device]", CUB_SMALL)
{
  using num_items_t = int;
  using value_t     = int;
  using offset_t    = cub::detail::choose_offset_t<num_items_t>;

  const num_items_t num_items = 2;
  auto d_in                   = cuda::constant_iterator(value_t{1});
  auto d_out                  = c2h::device_vector<value_t>(num_items);

  using selector_t = cub::detail::scan::
    policy_selector_from_types<decltype(d_in), decltype(d_out.begin()), value_t, offset_t, block_size_check_t>;

  cuda::compute_capability cc;
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc));
  const unsigned int target_block_size = selector_t{}(cc).lookback.threads_per_block;

  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  auto init = value_t{42};
  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveScan(d_in, d_out.begin(), block_size_check, init, num_items));
  REQUIRE(d_out[0] == init);
  REQUIRE(d_out[1] == (init + value_t{1}));

  // Make sure we use default tuning
  REQUIRE(d_block_size[0] == static_cast<unsigned int>(target_block_size));
}

CUB_TEST_CASE("Device scan exclusive scan with FutureValue works with default environment", "[scan][device]", CUB_SMALL)
{
  using num_items_t = int;

  const num_items_t num_items = 4;

  auto d_in  = c2h::device_vector<int>{1, 1, 1, 1};
  auto d_out = c2h::device_vector<int>(num_items);

  auto init_value_vec = c2h::device_vector<int>{42};
  auto future_init    = cub::FutureValue<int>(thrust::raw_pointer_cast(init_value_vec.data()));

  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveScan(d_in.begin(), d_out.begin(), cuda::std::plus{}, future_init, num_items));

  auto expected = c2h::device_vector<int>{42, 43, 44, 45};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device scan exclusive sum works with default environment", "[sum][device]", CUB_SMALL)
{
  using num_items_t = int;
  using value_t     = int;

  const num_items_t num_items = 2;

  auto d_in  = cuda::constant_iterator(value_t{1});
  auto d_out = c2h::device_vector<value_t>(num_items);

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(d_in, d_out.begin(), num_items));
  REQUIRE(d_out[0] == value_t{});
  REQUIRE(d_out[1] == value_t{} + d_in[0]);
}

CUB_TEST_CASE("Device scan inclusive-scan-init works with default environment", "[scan][device]", CUB_SMALL)
{
  using num_items_t = int;
  using value_t     = int;

  const num_items_t num_items = 3;
  auto d_in                   = cuda::constant_iterator(value_t{1});
  auto d_out                  = c2h::device_vector<value_t>(num_items);

  const value_t init{10};

  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveScanInit(d_in, d_out.begin(), cuda::std::plus{}, init, num_items));

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::make_counting_iterator(init + 1)));
}

#endif // TEST_LAUNCH == 0

#if TEST_LAUNCH != 1

template <int ThreadsPerBlock>
struct scan_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::ScanPolicy
  {
    return {cub::ScanAlgorithm::lookback,
            {ThreadsPerBlock,
             1,
             cub::BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE,
             cub::CacheLoadModifier::LOAD_DEFAULT,
             cub::BlockStoreAlgorithm::BLOCK_STORE_WARP_TRANSPOSE,
             cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING,
             cub::detail::default_delay_constructor_policy(true)},
            {}};
  }
};

struct unrelated_policy
{};

struct unrelated_tuning
{
  // should never be called
  auto operator()(cuda::compute_capability) const -> unrelated_policy
  {
    throw 1337;
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 32>, cuda::std::integral_constant<unsigned int, 64>>;

CUB_TEST("Device scan exclusive-scan can be tuned", "[scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  auto num_items = 3;
  auto d_in      = cuda::constant_iterator(1);
  auto d_out     = c2h::device_vector<int>(num_items);
  auto init      = int{42};

  // We are expecting that `unrelated_tuning` is ignored
  auto env = cuda::execution::tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveScan(d_in, d_out.begin(), block_size_check, init, num_items, env));

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::make_counting_iterator(init)));
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("Device scan exclusive-sum can be tuned", "[scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  auto num_items = target_block_size;
  c2h::device_vector<unsigned int> d_block_size(1, 0);
  // use block_size_recording_iterator to embed blockDim info in the input type and query after
  // since ExclusiveSum can not take a custom scan_op
  auto d_in  = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto d_out = c2h::device_vector<int>(num_items);

  // We are expecting that `unrelated_tuning` is ignored
  auto env = cuda::execution::tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(d_in, d_out.begin(), num_items, env));

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::counting_iterator<int>(0)));
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST_CASE("Device scan inclusive sum works with default environment", "[sum][device]", CUB_SMALL)
{
  using num_items_t = int;
  using value_t     = int;

  const num_items_t num_items = 3;

  auto d_in  = c2h::device_vector<value_t>{1, 1, 1};
  auto d_out = c2h::device_vector<value_t>(num_items);

  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveSum(d_in.begin(), d_out.begin(), num_items));

  auto expected = c2h::device_vector<value_t>{1, 2, 3};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device scan inclusive-scan works with default environment", "[scan][device]", CUB_SMALL)
{
  using num_items_t = int;
  using value_t     = int;
  using offset_t    = cub::detail::choose_offset_t<num_items_t>;

  const num_items_t num_items = 2;
  auto d_in                   = cuda::constant_iterator(value_t{1});
  auto d_out                  = c2h::device_vector<value_t>(num_items);

  using selector_t = cub::detail::scan::
    policy_selector_from_types<decltype(d_in), decltype(d_out.begin()), value_t, offset_t, block_size_check_t>;

  cuda::compute_capability cc;
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc));
  const unsigned int target_block_size = selector_t{}(cc).lookback.threads_per_block;

  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveScan(d_in, d_out.begin(), block_size_check, num_items));
  REQUIRE(d_out[0] == d_in[0]);
  REQUIRE(d_out[1] == d_in[0] + d_in[1]);

  // Make sure we use default tuning
  REQUIRE(d_block_size[0] == static_cast<unsigned int>(target_block_size));
}

CUB_TEST("Device scan inclusive-scan can be tuned", "[scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  auto num_items = 3;
  auto d_in      = cuda::constant_iterator(1);
  auto d_out     = c2h::device_vector<int>(num_items);

  // We are expecting that `unrelated_tuning` is ignored
  auto env = cuda::execution::tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveScan(d_in, d_out.begin(), block_size_check, num_items, env));

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::make_counting_iterator(1)));
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("Device scan inclusive-scan-init can be tuned", "[scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  auto num_items = 3;
  auto d_in      = cuda::constant_iterator(1);
  auto d_out     = c2h::device_vector<int>(num_items);

  const int init{10};

  // We are expecting that `unrelated_tuning` is ignored
  auto env = cuda::execution::tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  REQUIRE(
    cudaSuccess == cub::DeviceScan::InclusiveScanInit(d_in, d_out.begin(), block_size_check, init, num_items, env));

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::make_counting_iterator(init + 1)));
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST_CASE("Device scan exclusive sum in-place works with default environment", "[scan][device]", CUB_SMALL)
{
  auto d_data = c2h::device_vector<int>{1, 2, 3, 4};

  auto error = cub::DeviceScan::ExclusiveSum(d_data.begin(), static_cast<int>(d_data.size()));
  REQUIRE(error == cudaSuccess);

  auto expected = c2h::device_vector<int>{0, 1, 3, 6};
  REQUIRE(d_data == expected);
}

CUB_TEST_CASE("Device scan exclusive scan in-place works with default environment", "[scan][device]", CUB_SMALL)
{
  auto d_data = c2h::device_vector<int>{1, 2, 3, 4};

  auto error = cub::DeviceScan::ExclusiveScan(d_data.begin(), cuda::std::plus{}, 42, static_cast<int>(d_data.size()));
  REQUIRE(error == cudaSuccess);

  auto expected = c2h::device_vector<int>{42, 43, 45, 48};
  REQUIRE(d_data == expected);
}

CUB_TEST_CASE("Device scan inclusive sum in-place works with default environment", "[scan][device]", CUB_SMALL)
{
  auto d_data = c2h::device_vector<int>{1, 2, 3, 4};

  auto error = cub::DeviceScan::InclusiveSum(d_data.begin(), static_cast<int>(d_data.size()));
  REQUIRE(error == cudaSuccess);

  auto expected = c2h::device_vector<int>{1, 3, 6, 10};
  REQUIRE(d_data == expected);
}

CUB_TEST_CASE("Device scan inclusive scan in-place works with default environment", "[scan][device]", CUB_SMALL)
{
  auto d_data = c2h::device_vector<int>{1, 2, 3, 4};

  auto error = cub::DeviceScan::InclusiveScan(d_data.begin(), cuda::std::plus{}, static_cast<int>(d_data.size()));
  REQUIRE(error == cudaSuccess);

  auto expected = c2h::device_vector<int>{1, 3, 6, 10};
  REQUIRE(d_data == expected);
}

CUB_TEST_CASE("Device scan exclusive scan with FutureValue in-place works with default environment",
              "[scan][device]",
              CUB_SMALL)
{
  auto d_data = c2h::device_vector<int>{1, 2, 3, 4};

  auto init_value_vec = c2h::device_vector<int>{42};
  auto future_init    = cub::FutureValue<int>(thrust::raw_pointer_cast(init_value_vec.data()));

  auto error =
    cub::DeviceScan::ExclusiveScan(d_data.begin(), cuda::std::plus{}, future_init, static_cast<int>(d_data.size()));
  REQUIRE(error == cudaSuccess);

  auto expected = c2h::device_vector<int>{42, 43, 45, 48};
  REQUIRE(d_data == expected);
}

#endif // TEST_LAUNCH != 1

CUB_TEST("Device scan exclusive-scan uses environment", "[scan][device]", CUB_SMALL)
{
  using scan_op_t   = cuda::std::plus<>;
  using num_items_t = int;

  const num_items_t num_items = 10;
  auto d_in                   = cuda::constant_iterator(1.0f);
  auto d_out                  = c2h::device_vector<float>(num_items);

  using init_value_t = float;

  const init_value_t init{42.0f};

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveScan(
            nullptr, expected_bytes_allocated, d_in, d_out.begin(), scan_op_t{}, init, num_items));

  auto env = stdexec::env{cuda::execution::require(cuda::execution::determinism::not_guaranteed), // determinism
                          expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_scan_exclusive(d_in, d_out.begin(), scan_op_t{}, init, num_items, env);

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::counting_iterator<int>(static_cast<int>(init))));
}

CUB_TEST("Device scan exclusive-scan with FutureValue uses environment", "[scan][device]", CUB_SMALL)
{
  using scan_op_t   = cuda::std::plus<>;
  using num_items_t = int;

  const num_items_t num_items = 10;
  auto d_in                   = cuda::constant_iterator(1);
  auto d_out                  = c2h::device_vector<int>(num_items);

  auto init_value_vec = c2h::device_vector<int>{42};
  auto future_init    = cub::FutureValue<int>(thrust::raw_pointer_cast(init_value_vec.data()));

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveScan(
            nullptr, expected_bytes_allocated, d_in, d_out.begin(), scan_op_t{}, future_init, num_items));

  auto env = stdexec::env{cuda::execution::require(cuda::execution::determinism::not_guaranteed), // determinism
                          expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_scan_exclusive(d_in, d_out.begin(), scan_op_t{}, future_init, num_items, env);

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::counting_iterator<int>(42)));
}

CUB_TEST("Device scan exclusive-sum uses environment", "[scan][device]", CUB_SMALL)
{
  using scan_op_t   = cuda::std::plus<>;
  using num_items_t = int;

  const num_items_t num_items = 10;
  auto d_in                   = cuda::constant_iterator(1.0f);
  auto d_out                  = c2h::device_vector<float>(num_items);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess == cub::DeviceScan::ExclusiveSum(nullptr, expected_bytes_allocated, d_in, d_out.begin(), num_items));

  auto env = stdexec::env{cuda::execution::require(cuda::execution::determinism::not_guaranteed), // determinism
                          expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_scan_exclusive_sum(d_in, d_out.begin(), num_items, env);

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::counting_iterator<int>(0)));
}

CUB_TEST("Device scan inclusive-sum uses environment", "[scan][device]", CUB_SMALL)
{
  using num_items_t = int;

  const num_items_t num_items = GENERATE(0, 1, 10);
  auto d_in                   = c2h::device_vector<int>(num_items, 3);
  auto d_out                  = c2h::device_vector<int>(num_items);

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceScan::InclusiveSum(nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), num_items));

  auto env = stdexec::env{cuda::execution::require(cuda::execution::determinism::not_guaranteed), // determinism
                          expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_scan_inclusive_sum(d_in.begin(), d_out.begin(), num_items, env);

  thrust::host_vector<int> h_expected(num_items);
  for (int i = 0; i < num_items; i++)
  {
    h_expected[i] = 3 * (i + 1);
  }
  const c2h::device_vector<int> expected = h_expected;
  REQUIRE(d_out == expected);
}

CUB_TEST("Device scan inclusive-scan uses environment", "[scan][device]", CUB_SMALL)
{
  using scan_op_t   = cuda::std::plus<>;
  using num_items_t = int;

  const num_items_t num_items = 10;
  auto d_in                   = cuda::constant_iterator(1.0f);
  auto d_out                  = c2h::device_vector<float>(num_items);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveScan(nullptr, expected_bytes_allocated, d_in, d_out.begin(), scan_op_t{}, num_items));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_scan_inclusive(d_in, d_out.begin(), scan_op_t{}, num_items, env);

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::counting_iterator<int>(1)));
}

CUB_TEST("Device scan inclusive-scan-init uses environment", "[scan][device]", CUB_SMALL)
{
  using num_items_t = int;

  const num_items_t num_items = 4;
  auto d_in                   = c2h::device_vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
  auto d_out                  = c2h::device_vector<float>(num_items);
  const float init            = 10.0f;

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceScan::InclusiveScanInit(
            nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), cuda::std::plus{}, init, num_items));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_scan_inclusive_init(d_in.begin(), d_out.begin(), cuda::std::plus{}, init, num_items, env);

  auto expected = c2h::device_vector<float>{11.0f, 13.0f, 16.0f, 20.0f};
  REQUIRE(d_out == expected);
}

CUB_TEST("Device scan exclusive-sum in-place uses environment", "[scan][device]", CUB_SMALL)
{
  auto d_data = c2h::device_vector<int>{1, 2, 3, 4};

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveSum(
            nullptr, expected_bytes_allocated, d_data.begin(), d_data.begin(), static_cast<int>(d_data.size())));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_scan_exclusive_sum(d_data.begin(), static_cast<int>(d_data.size()), env);

  auto expected = c2h::device_vector<int>{0, 1, 3, 6};
  REQUIRE(d_data == expected);
}

CUB_TEST("Device scan exclusive-scan in-place uses environment", "[scan][device]", CUB_SMALL)
{
  auto d_data = c2h::device_vector<int>{1, 2, 3, 4};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveScan(
      nullptr,
      expected_bytes_allocated,
      d_data.begin(),
      d_data.begin(),
      cuda::std::plus{},
      42,
      static_cast<int>(d_data.size())));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_scan_exclusive(d_data.begin(), cuda::std::plus{}, 42, static_cast<int>(d_data.size()), env);

  auto expected = c2h::device_vector<int>{42, 43, 45, 48};
  REQUIRE(d_data == expected);
}

CUB_TEST("Device scan exclusive-scan with FutureValue in-place uses environment", "[scan][device]", CUB_SMALL)
{
  auto d_data = c2h::device_vector<int>{1, 1, 1, 1};

  auto init_value_vec = c2h::device_vector<int>{42};
  auto future_init    = cub::FutureValue<int>(thrust::raw_pointer_cast(init_value_vec.data()));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveScan(
      nullptr,
      expected_bytes_allocated,
      d_data.begin(),
      d_data.begin(),
      cuda::std::plus{},
      future_init,
      static_cast<int>(d_data.size())));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_scan_exclusive(d_data.begin(), cuda::std::plus{}, future_init, static_cast<int>(d_data.size()), env);

  auto expected = c2h::device_vector<int>{42, 43, 44, 45};
  REQUIRE(d_data == expected);
}

CUB_TEST("Device scan inclusive-sum in-place uses environment", "[scan][device]", CUB_SMALL)
{
  auto d_data = c2h::device_vector<int>{1, 2, 3, 4};

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceScan::InclusiveSum(
            nullptr, expected_bytes_allocated, d_data.begin(), d_data.begin(), static_cast<int>(d_data.size())));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_scan_inclusive_sum(d_data.begin(), static_cast<int>(d_data.size()), env);

  auto expected = c2h::device_vector<int>{1, 3, 6, 10};
  REQUIRE(d_data == expected);
}

CUB_TEST("Device scan inclusive-scan in-place uses environment", "[scan][device]", CUB_SMALL)
{
  auto d_data = c2h::device_vector<int>{1, 2, 3, 4};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveScan(
      nullptr,
      expected_bytes_allocated,
      d_data.begin(),
      d_data.begin(),
      cuda::std::plus{},
      static_cast<int>(d_data.size())));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_scan_inclusive(d_data.begin(), cuda::std::plus{}, static_cast<int>(d_data.size()), env);

  auto expected = c2h::device_vector<int>{1, 3, 6, 10};
  REQUIRE(d_data == expected);
}

#if TEST_LAUNCH == 0

// The two-phase overloads take the same environment as the single-phase ones but never allocate, so they do not go
// through the launch wrappers and would run identically in every TEST_LAUNCH variant. Test them with host launch only.

// Runs the two-phase overload wrapped by two_phase: queries the temporary storage size and executes with user provided
// storage.
template <class TwoPhaseFn>
void run_two_phase(TwoPhaseFn two_phase)
{
  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == two_phase(nullptr, temp_storage_bytes));

  c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  REQUIRE(cudaSuccess == two_phase(thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

CUB_TEST_CASE("Device scan exclusive-sum works with user provided memory and environment", "[scan][device]", CUB_SMALL)
{
  const auto d_in = c2h::device_vector<int>{1, 2, 3, 4};
  auto d_out      = c2h::device_vector<int>(d_in.size(), thrust::no_init);
  const int num_items{static_cast<int>(d_in.size())};

  test_with_custom_streams([&](const auto& env) {
    run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_items, env);
    });

    const auto expected = c2h::device_vector<int>{0, 1, 3, 6};
    REQUIRE(d_out == expected);
  });
}

CUB_TEST_CASE("Device scan exclusive-sum in-place works with user provided memory and environment",
              "[scan][device]",
              CUB_SMALL)
{
  auto d_data = c2h::device_vector<int>{1, 2, 3, 4};
  const int num_items{static_cast<int>(d_data.size())};

  test_with_custom_streams([&](const auto& env) {
    run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_data.begin(), num_items, env);
    });

    const auto expected = c2h::device_vector<int>{0, 1, 3, 6};
    REQUIRE(d_data == expected);
  });
}

CUB_TEST_CASE("Device scan exclusive-scan works with user provided memory and environment", "[scan][device]", CUB_SMALL)
{
  const auto d_in = c2h::device_vector<int>{1, 2, 3, 4};
  auto d_out      = c2h::device_vector<int>(d_in.size(), thrust::no_init);
  const int num_items{static_cast<int>(d_in.size())};

  test_with_custom_streams([&](const auto& env) {
    run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::ExclusiveScan(
        d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), cuda::std::plus<>{}, 10, num_items, env);
    });

    const auto expected = c2h::device_vector<int>{10, 11, 13, 16};
    REQUIRE(d_out == expected);
  });
}

CUB_TEST_CASE("Device scan exclusive-scan in-place works with user provided memory and environment",
              "[scan][device]",
              CUB_SMALL)
{
  auto d_data = c2h::device_vector<int>{1, 2, 3, 4};
  const int num_items{static_cast<int>(d_data.size())};

  test_with_custom_streams([&](const auto& env) {
    run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::ExclusiveScan(
        d_temp_storage, temp_storage_bytes, d_data.begin(), cuda::std::plus<>{}, 10, num_items, env);
    });

    const auto expected = c2h::device_vector<int>{10, 11, 13, 16};
    REQUIRE(d_data == expected);
  });
}

CUB_TEST_CASE("Device scan exclusive-scan with FutureValue works with user provided memory and environment",
              "[scan][device]",
              CUB_SMALL)
{
  const auto d_in = c2h::device_vector<int>{1, 2, 3, 4};
  auto d_out      = c2h::device_vector<int>(d_in.size(), thrust::no_init);
  auto d_init     = c2h::device_vector<int>{10};
  const int num_items{static_cast<int>(d_in.size())};

  test_with_custom_streams([&](const auto& env) {
    run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::ExclusiveScan(
        d_temp_storage,
        temp_storage_bytes,
        d_in.begin(),
        d_out.begin(),
        cuda::std::plus<>{},
        cub::FutureValue<int>(thrust::raw_pointer_cast(d_init.data())),
        num_items,
        env);
    });

    const auto expected = c2h::device_vector<int>{10, 11, 13, 16};
    REQUIRE(d_out == expected);
  });
}

CUB_TEST_CASE("Device scan exclusive-scan with FutureValue in-place works with user provided memory and environment",
              "[scan][device]",
              CUB_SMALL)
{
  auto d_data = c2h::device_vector<int>{1, 2, 3, 4};
  auto d_init = c2h::device_vector<int>{10};
  const int num_items{static_cast<int>(d_data.size())};

  test_with_custom_streams([&](const auto& env) {
    run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::ExclusiveScan(
        d_temp_storage,
        temp_storage_bytes,
        d_data.begin(),
        cuda::std::plus<>{},
        cub::FutureValue<int>(thrust::raw_pointer_cast(d_init.data())),
        num_items,
        env);
    });

    const auto expected = c2h::device_vector<int>{10, 11, 13, 16};
    REQUIRE(d_data == expected);
  });
}

CUB_TEST_CASE("Device scan inclusive-sum works with user provided memory and environment", "[scan][device]", CUB_SMALL)
{
  const auto d_in = c2h::device_vector<int>{1, 2, 3, 4};
  auto d_out      = c2h::device_vector<int>(d_in.size(), thrust::no_init);
  const int num_items{static_cast<int>(d_in.size())};

  test_with_custom_streams([&](const auto& env) {
    run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_items, env);
    });

    const auto expected = c2h::device_vector<int>{1, 3, 6, 10};
    REQUIRE(d_out == expected);
  });
}

CUB_TEST_CASE("Device scan inclusive-sum in-place works with user provided memory and environment",
              "[scan][device]",
              CUB_SMALL)
{
  auto d_data = c2h::device_vector<int>{1, 2, 3, 4};
  const int num_items{static_cast<int>(d_data.size())};

  test_with_custom_streams([&](const auto& env) {
    run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_data.begin(), num_items, env);
    });

    const auto expected = c2h::device_vector<int>{1, 3, 6, 10};
    REQUIRE(d_data == expected);
  });
}

CUB_TEST_CASE("Device scan inclusive-scan works with user provided memory and environment", "[scan][device]", CUB_SMALL)
{
  const auto d_in = c2h::device_vector<int>{1, 2, 3, 4};
  auto d_out      = c2h::device_vector<int>(d_in.size(), thrust::no_init);
  const int num_items{static_cast<int>(d_in.size())};

  test_with_custom_streams([&](const auto& env) {
    run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::InclusiveScan(
        d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), cuda::std::plus<>{}, num_items, env);
    });

    const auto expected = c2h::device_vector<int>{1, 3, 6, 10};
    REQUIRE(d_out == expected);
  });
}

CUB_TEST_CASE("Device scan inclusive-scan in-place works with user provided memory and environment",
              "[scan][device]",
              CUB_SMALL)
{
  auto d_data = c2h::device_vector<int>{1, 2, 3, 4};
  const int num_items{static_cast<int>(d_data.size())};

  test_with_custom_streams([&](const auto& env) {
    run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::InclusiveScan(
        d_temp_storage, temp_storage_bytes, d_data.begin(), cuda::std::plus<>{}, num_items, env);
    });

    const auto expected = c2h::device_vector<int>{1, 3, 6, 10};
    REQUIRE(d_data == expected);
  });
}

CUB_TEST_CASE("Device scan inclusive-scan-init works with user provided memory and environment",
              "[scan][device]",
              CUB_SMALL)
{
  const auto d_in = c2h::device_vector<int>{1, 2, 3, 4};
  auto d_out      = c2h::device_vector<int>(d_in.size(), thrust::no_init);
  const int num_items{static_cast<int>(d_in.size())};

  test_with_custom_streams([&](const auto& env) {
    run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::InclusiveScanInit(
        d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), cuda::std::plus<>{}, 10, num_items, env);
    });

    const auto expected = c2h::device_vector<int>{11, 13, 16, 20};
    REQUIRE(d_out == expected);
  });
}

CUB_TEST_CASE("Device scan inclusive-scan-init with deferred init works with user provided memory and environment",
              "[scan][device]",
              CUB_SMALL)
{
  const auto d_in   = c2h::device_vector<int>{1, 2, 3, 4};
  auto d_out        = c2h::device_vector<int>(d_in.size(), thrust::no_init);
  const auto d_init = c2h::device_vector<int>{10};
  const int num_items{static_cast<int>(d_in.size())};

  test_with_custom_streams([&](const auto& env) {
    run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::InclusiveScanInit(
        d_temp_storage,
        temp_storage_bytes,
        d_in.begin(),
        d_out.begin(),
        cuda::std::plus<>{},
        cuda::args::deferred(d_init.begin()),
        num_items,
        env);
    });

    const auto expected = c2h::device_vector<int>{11, 13, 16, 20};
    REQUIRE(d_out == expected);
  });
}

// Before the environment parameter, the two-phase overloads took `cudaStream_t stream = nullptr`, so callers passing
// nullptr or a literal 0 for the stream exist. Both must keep compiling and keep running on the default stream.
CUB_TEST_CASE("Device scan two-phase overloads accept legacy null stream arguments", "[scan][device]", CUB_SMALL)
{
  const auto d_in = c2h::device_vector<int>{1, 2, 3, 4};
  auto d_out      = c2h::device_vector<int>(d_in.size(), thrust::no_init);
  const int num_items{static_cast<int>(d_in.size())};

  auto exclusive_sum_on = [&](const auto& stream) {
    const stream_scope scope{cudaStream_t{}};
    run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), num_items, stream);
    });
  };

  auto exclusive_scan_on = [&](const auto& stream) {
    const stream_scope scope{cudaStream_t{}};
    run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceScan::ExclusiveScan(
        d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), cuda::std::plus<>{}, 0, num_items, stream);
    });
  };

  SECTION("exclusive-sum with nullptr")
  {
    exclusive_sum_on(nullptr);
  }

  SECTION("exclusive-sum with literal 0")
  {
    exclusive_sum_on(0);
  }

  SECTION("exclusive-scan with nullptr")
  {
    exclusive_scan_on(nullptr);
  }

  SECTION("exclusive-scan with literal 0")
  {
    exclusive_scan_on(0);
  }

  const auto expected = c2h::device_vector<int>{0, 1, 3, 6};
  REQUIRE(d_out == expected);
}

CUB_TEST_CASE("Device scan in-place two-phase overloads accept a legacy nullptr stream", "[scan][device]", CUB_SMALL)
{
  auto d_data = c2h::device_vector<int>{1, 2, 3, 4};
  const int num_items{static_cast<int>(d_data.size())};

  const stream_scope scope{cudaStream_t{}};
  run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
    return cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_data.begin(), num_items, nullptr);
  });

  const auto expected = c2h::device_vector<int>{1, 3, 6, 10};
  REQUIRE(d_data == expected);
}

CUB_TEST("Device scan exclusive-scan can be tuned with user provided memory", "[scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  const int num_items = 3;
  auto d_in           = cuda::constant_iterator(1);
  auto d_out          = c2h::device_vector<int>(num_items, thrust::no_init);
  const int init{42};

  // We are expecting that `unrelated_tuning` is ignored
  const auto env = cuda::execution::tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
    return cub::DeviceScan::ExclusiveScan(
      d_temp_storage, temp_storage_bytes, d_in, d_out.begin(), block_size_check, init, num_items, env);
  });

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::make_counting_iterator(init)));
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("Device scan exclusive-sum can be tuned with user provided memory", "[scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  const auto num_items = static_cast<int>(target_block_size);
  c2h::device_vector<unsigned int> d_block_size(1, 0);
  // ExclusiveSum can not take a custom scan_op, so the input iterator records the block size
  auto d_in  = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto d_out = c2h::device_vector<int>(num_items, thrust::no_init);

  // We are expecting that `unrelated_tuning` is ignored
  const auto env = cuda::execution::tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
    return cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out.begin(), num_items, env);
  });

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::counting_iterator<int>(0)));
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("Device scan inclusive-scan can be tuned with user provided memory", "[scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  const int num_items = 3;
  auto d_in           = cuda::constant_iterator(1);
  auto d_out          = c2h::device_vector<int>(num_items, thrust::no_init);

  // We are expecting that `unrelated_tuning` is ignored
  const auto env = cuda::execution::tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
    return cub::DeviceScan::InclusiveScan(
      d_temp_storage, temp_storage_bytes, d_in, d_out.begin(), block_size_check, num_items, env);
  });

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::make_counting_iterator(1)));
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST(
  "Device scan inclusive-scan-init can be tuned with user provided memory", "[scan][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  const int num_items = 3;
  auto d_in           = cuda::constant_iterator(1);
  auto d_out          = c2h::device_vector<int>(num_items, thrust::no_init);
  const int init{10};

  // We are expecting that `unrelated_tuning` is ignored
  const auto env = cuda::execution::tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
    return cub::DeviceScan::InclusiveScanInit(
      d_temp_storage, temp_storage_bytes, d_in, d_out.begin(), block_size_check, init, num_items, env);
  });

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::make_counting_iterator(init + 1)));
  REQUIRE(d_block_size[0] == target_block_size);
}

// The two-phase overloads read the determinism requirement from the environment like the single-phase ones: a
// run_to_run floating-point sum selects the stable reduction order, so both overloads produce bitwise equal results.
CUB_TEST_CASE("Device scan exclusive-scan with user provided memory honors determinism requirements",
              "[scan][device]",
              CUB_SMALL)
{
  const int num_items = 10'000;
  c2h::device_vector<float> d_in(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), d_in, -100.0f, 100.0f);

  const auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);
  const auto op  = cuda::std::plus<float>{};

  c2h::device_vector<float> d_reference(num_items, thrust::no_init);
  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveScan(d_in.begin(), d_reference.begin(), op, 0.0f, num_items, env));

  c2h::device_vector<float> d_out(num_items, thrust::no_init);
  run_two_phase([&](void* d_temp_storage, size_t& temp_storage_bytes) {
    return cub::DeviceScan::ExclusiveScan(
      d_temp_storage, temp_storage_bytes, d_in.begin(), d_out.begin(), op, 0.0f, num_items, env);
  });

  REQUIRE_THAT(detail::to_vec(d_reference), detail::BitwiseEqualsRange(detail::to_vec(d_out)));
}

#endif // TEST_LAUNCH == 0

#if _CCCL_COMPILER(GCC, >=, 8) // gcc 7 cannot preserve constexpr-ness from p1 to p2
CUB_TEST("Test ScanPolicy properties", "[scan][device]", CUB_SMALL)
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::ScanPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::ScanPolicy>);

  STATIC_REQUIRE(::cuda::std::semiregular<cub::ScanLookbackPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::ScanLookbackPolicy>);

  STATIC_REQUIRE(::cuda::std::semiregular<cub::ScanLookaheadPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::ScanLookaheadPolicy>);

  // aggregate init
  constexpr auto p1_lb = cub::ScanLookbackPolicy{
    256,
    11,
    cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
    cub::CacheLoadModifier::LOAD_DEFAULT,
    cub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT,
    cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING,
    cub::LookbackDelayPolicy{cub::LookbackDelayAlgorithm::fixed_delay, 832, 1165}};
  constexpr auto p1_la = cub::ScanLookaheadPolicy{3, 8, 4, 2};
  constexpr auto p1    = cub::ScanPolicy{cub::ScanAlgorithm::lookback, p1_lb, p1_la};

#  if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto p2_lb = cub::ScanLookbackPolicy{
    .threads_per_block = 256,
    .items_per_thread  = 11,
    .load_algorithm    = cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT,
    .load_modifier     = cub::CacheLoadModifier::LOAD_DEFAULT,
    .store_algorithm   = cub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT,
    .scan_algorithm    = cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING,
    .lookback_delay    = cub::LookbackDelayPolicy{
      .kind = cub::LookbackDelayAlgorithm::fixed_delay, .delay = 832, .l2_write_latency = 1165}};
  constexpr auto p2_la = cub::ScanLookaheadPolicy{
    .reduce_and_scan_warps = 3, .items_per_thread = 8, .lookahead_items_per_thread = 4, .lookahead_stages = 2};

  constexpr auto p2 = cub::ScanPolicy{.algorithm = cub::ScanAlgorithm::lookback, .lookback = p2_lb, .lookahead = p2_la};
#  else // _CCCL_STD_VER >= 2020
  constexpr auto p2_lb = p1_lb;
  constexpr auto p2_la = p1_la;
  constexpr auto p2    = p1;
#  endif // _CCCL_STD_VER >= 2020

  // comparison
  STATIC_REQUIRE(p1_lb == p2_lb);
  STATIC_REQUIRE_FALSE(p1_lb != p2_lb);

  STATIC_REQUIRE(p1_la == p2_la);
  STATIC_REQUIRE_FALSE(p1_la != p2_la);

  STATIC_REQUIRE(p1 == p2);
  STATIC_REQUIRE_FALSE(p1 != p2);

  auto to_string = [](const auto& p) {
    std::ostringstream os;
    os << p;
    return os.str();
  };
  REQUIRE(to_string(p1_lb)
          == "ScanLookbackPolicy { .threads_per_block = 256, .items_per_thread = 11"
             ", .load_algorithm = BLOCK_LOAD_DIRECT, .load_modifier = LOAD_DEFAULT"
             ", .store_algorithm = BLOCK_STORE_DIRECT, .scan_algorithm = BLOCK_SCAN_RAKING"
             ", .lookback_delay = LookbackDelayPolicy { .kind = LookbackDelayAlgorithm::fixed_delay"
             ", .delay = 832, .l2_write_latency = 1165 } }");
  REQUIRE(to_string(p1_la)
          == "ScanLookaheadPolicy { .reduce_and_scan_warps = 3, .items_per_thread = 8"
             ", .lookahead_items_per_thread = 4, .lookahead_stages = 2, .block_idx_stages = -1 }");
  REQUIRE(
    to_string(p1)
    == "ScanPolicy { .algorithm = ScanAlgorithm::lookback"
       ", .lookback = ScanLookbackPolicy { .threads_per_block = 256, .items_per_thread = 11"
       ", .load_algorithm = BLOCK_LOAD_DIRECT, .load_modifier = LOAD_DEFAULT"
       ", .store_algorithm = BLOCK_STORE_DIRECT, .scan_algorithm = BLOCK_SCAN_RAKING"
       ", .lookback_delay = LookbackDelayPolicy { .kind = LookbackDelayAlgorithm::fixed_delay"
       ", .delay = 832, .l2_write_latency = 1165 } }"
       ", .lookahead = ScanLookaheadPolicy { .reduce_and_scan_warps = 3, .items_per_thread = 8"
       ", .lookahead_items_per_thread = 4, .lookahead_stages = 2, .block_idx_stages = -1 } }");
}
#endif // _CCCL_COMPILER(GCC, >=, 8)
