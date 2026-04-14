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

#include <cuda/__device/arch_id.h>
#include <cuda/iterator>

#include "catch2_test_device_scan.cuh"
#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveScan, device_scan_exclusive);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveSum, device_scan_exclusive_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScan, device_scan_inclusive);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveSum, device_scan_inclusive_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScanInit, device_scan_inclusive_init);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

// Launcher helper always passes an environment.
// We need a test of simple use to check if default environment works.
// ifdef it out not to spend time compiling and running it twice.
#if TEST_LAUNCH == 0
struct block_size_check_t
{
  int* ptr;

  __device__ int operator()(int a, int b)
  {
    if (threadIdx.x == 0)
    {
      *ptr = blockDim.x;
    }
    return a + b;
  }
};

TEST_CASE("Device scan exclusive scan works with default environment", "[scan][device]")
{
  using num_items_t = int;
  using value_t     = int;
  using offset_t    = cub::detail::choose_offset_t<num_items_t>;

  num_items_t num_items = 2;
  auto d_in             = cuda::constant_iterator(value_t{1});
  auto d_out            = c2h::device_vector<value_t>(num_items);

  using selector_t = cub::detail::scan::
    policy_selector_from_types<decltype(d_in), decltype(d_out.begin()), value_t, offset_t, block_size_check_t>;

  cuda::arch_id arch_id;
  REQUIRE(cudaSuccess == cub::detail::ptx_arch_id(arch_id));
  const auto target_block_size = selector_t{}(arch_id).lookback.block_threads;

  c2h::device_vector<int> d_block_size(1);
  block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  auto init = value_t{42};
  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveScan(d_in, d_out.begin(), block_size_check, init, num_items));
  REQUIRE(d_out[0] == init);
  REQUIRE(d_out[1] == (init + value_t{1}));

  // Make sure we use default tuning
  REQUIRE(d_block_size[0] == target_block_size);
}

TEST_CASE("Device scan exclusive scan with FutureValue works with default environment", "[scan][device]")
{
  using num_items_t = int;

  num_items_t num_items = 4;

  auto d_in  = c2h::device_vector<int>{1, 1, 1, 1};
  auto d_out = c2h::device_vector<int>(num_items);

  auto init_value_vec = c2h::device_vector<int>{42};
  auto future_init    = cub::FutureValue<int>(thrust::raw_pointer_cast(init_value_vec.data()));

  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveScan(d_in.begin(), d_out.begin(), cuda::std::plus{}, future_init, num_items));

  auto expected = c2h::device_vector<int>{42, 43, 44, 45};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device scan exclusive sum works with default environment", "[sum][device]")
{
  using num_items_t = int;
  using value_t     = int;

  num_items_t num_items = 2;

  auto d_in  = cuda::constant_iterator(value_t{1});
  auto d_out = c2h::device_vector<value_t>(num_items);

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(d_in, d_out.begin(), num_items));
  REQUIRE(d_out[0] == value_t{});
  REQUIRE(d_out[1] == value_t{} + d_in[0]);
}

template <int BlockThreads>
struct scan_tuning
{
  _CCCL_API constexpr auto operator()(cuda::arch_id /*arch*/) const -> cub::detail::scan::scan_policy
  {
    return {cub::detail::scan::scan_algorithm::lookback,
            {BlockThreads,
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
  auto operator()(cuda::arch_id /*arch*/) const -> unrelated_policy
  {
    throw 1337;
  }
};

using block_sizes = c2h::type_list<cuda::std::integral_constant<int, 32>, cuda::std::integral_constant<int, 64>>;

C2H_TEST("Device scan exclusive-scan can be tuned", "[scan][device]", block_sizes)
{
  constexpr int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_block_size(1);
  block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  auto num_items = 3;
  auto d_in      = cuda::constant_iterator(1);
  auto d_out     = c2h::device_vector<int>(num_items);
  auto init      = int{42};

  // We are expecting that `unrelated_tuning` is ignored
  auto env = cuda::execution::__tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveScan(d_in, d_out.begin(), block_size_check, init, num_items, env));

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::make_counting_iterator(init)));
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Device scan exclusive-sum can be tuned", "[scan][device]", block_sizes)
{
  constexpr int target_block_size = c2h::get<0, TestType>::value;

  auto num_items = target_block_size;
  c2h::device_vector<int> d_block_size(1, 0);
  // use block_size_recording_iterator to embed blockDim info in the input type and query after
  // since ExclusiveSum can not take a custom scan_op
  auto d_in  = block_size_recording_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto d_out = c2h::device_vector<int>(num_items);

  // We are expecting that `unrelated_tuning` is ignored
  auto env = cuda::execution::__tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(d_in, d_out.begin(), num_items, env));

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::counting_iterator<int>(0)));
  REQUIRE(d_block_size[0] == target_block_size);
}

TEST_CASE("Device scan inclusive sum works with default environment", "[sum][device]")
{
  using num_items_t = int;
  using value_t     = int;

  num_items_t num_items = 3;

  auto d_in  = c2h::device_vector<value_t>{1, 1, 1};
  auto d_out = c2h::device_vector<value_t>(num_items);

  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveSum(d_in.begin(), d_out.begin(), num_items));

  auto expected = c2h::device_vector<value_t>{1, 2, 3};
  REQUIRE(d_out == expected);
}

TEST_CASE("Device scan inclusive-scan works with default environment", "[scan][device]")
{
  using num_items_t = int;
  using value_t     = int;
  using offset_t    = cub::detail::choose_offset_t<num_items_t>;

  num_items_t num_items = 2;
  auto d_in             = cuda::constant_iterator(value_t{1});
  auto d_out            = c2h::device_vector<value_t>(num_items);

  using selector_t = cub::detail::scan::
    policy_selector_from_types<decltype(d_in), decltype(d_out.begin()), value_t, offset_t, block_size_check_t>;

  cuda::arch_id arch_id;
  REQUIRE(cudaSuccess == cub::detail::ptx_arch_id(arch_id));
  const auto target_block_size = selector_t{}(arch_id).lookback.block_threads;

  c2h::device_vector<int> d_block_size(1);
  block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveScan(d_in, d_out.begin(), block_size_check, num_items));
  REQUIRE(d_out[0] == d_in[0]);
  REQUIRE(d_out[1] == d_in[0] + d_in[1]);

  // Make sure we use default tuning
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Device scan inclusive-scan can be tuned", "[scan][device]", block_sizes)
{
  constexpr int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_block_size(1);
  block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  auto num_items = 3;
  auto d_in      = cuda::constant_iterator(1);
  auto d_out     = c2h::device_vector<int>(num_items);

  // We are expecting that `unrelated_tuning` is ignored
  auto env = cuda::execution::__tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveScan(d_in, d_out.begin(), block_size_check, num_items, env));

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::make_counting_iterator(1)));
  REQUIRE(d_block_size[0] == target_block_size);
}

TEST_CASE("Device scan inclusive-scan-init works with default environment", "[scan][device]")
{
  using num_items_t = int;
  using value_t     = int;

  num_items_t num_items = 3;
  auto d_in             = cuda::constant_iterator(value_t{1});
  auto d_out            = c2h::device_vector<value_t>(num_items);

  value_t init{10};

  REQUIRE(cudaSuccess == cub::DeviceScan::InclusiveScanInit(d_in, d_out.begin(), cuda::std::plus{}, init, num_items));

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::make_counting_iterator(init + 1)));
}

C2H_TEST("Device scan inclusive-scan-init can be tuned", "[scan][device]", block_sizes)
{
  constexpr int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_block_size(1);
  block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  auto num_items = 3;
  auto d_in      = cuda::constant_iterator(1);
  auto d_out     = c2h::device_vector<int>(num_items);

  int init{10};

  // We are expecting that `unrelated_tuning` is ignored
  auto env = cuda::execution::__tune(scan_tuning<target_block_size>{}, unrelated_tuning{});

  REQUIRE(
    cudaSuccess == cub::DeviceScan::InclusiveScanInit(d_in, d_out.begin(), block_size_check, init, num_items, env));

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::make_counting_iterator(init + 1)));
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif

C2H_TEST("Device scan exclusive-scan uses environment", "[scan][device]")
{
  using scan_op_t   = cuda::std::plus<>;
  using num_items_t = int;

  num_items_t num_items = 10;
  auto d_in             = cuda::constant_iterator(1.0f);
  auto d_out            = c2h::device_vector<float>(num_items);

  using init_t = float;

  init_t init{42.0f};

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceScan::ExclusiveScan(
            nullptr, expected_bytes_allocated, d_in, d_out.begin(), scan_op_t{}, init, num_items));

  auto env = stdexec::env{cuda::execution::require(cuda::execution::determinism::not_guaranteed), // determinism
                          expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_scan_exclusive(d_in, d_out.begin(), scan_op_t{}, init, num_items, env);

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::counting_iterator<int>(static_cast<int>(init))));
}

C2H_TEST("Device scan exclusive-scan with FutureValue uses environment", "[scan][device]")
{
  using scan_op_t   = cuda::std::plus<>;
  using num_items_t = int;

  num_items_t num_items = 10;
  auto d_in             = cuda::constant_iterator(1);
  auto d_out            = c2h::device_vector<int>(num_items);

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

C2H_TEST("Device scan exclusive-sum uses environment", "[scan][device]")
{
  using scan_op_t   = cuda::std::plus<>;
  using num_items_t = int;

  num_items_t num_items = 10;
  auto d_in             = cuda::constant_iterator(1.0f);
  auto d_out            = c2h::device_vector<float>(num_items);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess == cub::DeviceScan::ExclusiveSum(nullptr, expected_bytes_allocated, d_in, d_out.begin(), num_items));

  auto env = stdexec::env{cuda::execution::require(cuda::execution::determinism::not_guaranteed), // determinism
                          expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_scan_exclusive_sum(d_in, d_out.begin(), num_items, env);

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::counting_iterator<int>(0)));
}

C2H_TEST("Device scan inclusive-sum uses environment", "[scan][device]")
{
  using num_items_t = int;

  num_items_t num_items = GENERATE(0, 1, 10);
  auto d_in             = c2h::device_vector<int>(num_items, 3);
  auto d_out            = c2h::device_vector<int>(num_items);

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
  c2h::device_vector<int> expected = h_expected;
  REQUIRE(d_out == expected);
}

C2H_TEST("Device scan inclusive-scan uses environment", "[scan][device]")
{
  using scan_op_t   = cuda::std::plus<>;
  using num_items_t = int;

  num_items_t num_items = 10;
  auto d_in             = cuda::constant_iterator(1.0f);
  auto d_out            = c2h::device_vector<float>(num_items);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::InclusiveScan(nullptr, expected_bytes_allocated, d_in, d_out.begin(), scan_op_t{}, num_items));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_scan_inclusive(d_in, d_out.begin(), scan_op_t{}, num_items, env);

  REQUIRE(thrust::equal(d_out.begin(), d_out.end(), thrust::counting_iterator<int>(1)));
}

C2H_TEST("Device scan inclusive-scan-init uses environment", "[scan][device]")
{
  using num_items_t = int;

  num_items_t num_items = 4;
  auto d_in             = c2h::device_vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
  auto d_out            = c2h::device_vector<float>(num_items);
  float init            = 10.0f;

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceScan::InclusiveScanInit(
            nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), cuda::std::plus{}, init, num_items));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_scan_inclusive_init(d_in.begin(), d_out.begin(), cuda::std::plus{}, init, num_items, env);

  auto expected = c2h::device_vector<float>{11.0f, 13.0f, 16.0f, 20.0f};
  REQUIRE(d_out == expected);
}
