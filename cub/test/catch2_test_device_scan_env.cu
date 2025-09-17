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
#include <thrust/iterator/constant_iterator.h>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveScan, device_scan_exclusive);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveSum, device_scan_exclusive_sum);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

// Launcher helper always passes an environment.
// We need a test of simple use to check if default environment works.
// ifdef it out not to spend time compiling and running it twice.
// #if TEST_LAUNCH == 0
#if 0
struct block_size_check_t
{
  int* ptr;

  __device__ int operator()(int a, int b)
  {
    *ptr = blockDim.x;
    return a + b;
  }
};

struct block_size_retreiver_t
{
  int* ptr;

  template <class ActivePolicyT>
  cudaError_t Invoke()
  {
    *ptr = ActivePolicyT::ScanPolicyT::BLOCK_THREADS;
    return cudaSuccess;
  }
};

TEST_CASE("Device scan exclusive scan works with default environment", "[scan][device]")
{
  using num_items_t = int;
  using value_t     = int;
  using offset_t    = cub::detail::choose_offset_t<num_items_t>;

  using policy_t =
    cub::detail::scan::default_tuning::fn<value_t, value_t, value_t, offset_t, block_size_check_t>::MaxPolicy;

  int current_device{};
  REQUIRE(cudaSuccess == cudaGetDevice(&current_device));

  int ptx_version{};
  REQUIRE(cudaSuccess == cub::PtxVersion(ptx_version, current_device));

  int target_block_size{};
  block_size_retreiver_t block_size_retreiver{&target_block_size};
  REQUIRE(cudaSuccess == policy_t::Invoke(ptx_version, block_size_retreiver));

  num_items_t num_items = 1;
  c2h::device_vector<int> d_block_size(1);
  block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};
  auto d_in  = thrust::make_constant_iterator(value_t{1});
  auto d_out = thrust::device_vector<value_t>(1);

  auto init = value_t{0};
  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveScan(d_in, d_out.begin(), block_size_check, init, num_items));
  REQUIRE(d_out[0] == init);

  // Make sure we use default tuning
  REQUIRE(d_block_size[0] == target_block_size);
}

TEST_CASE("Device scan exclusive sum works with default environment", "[sum][device]")
{
  using num_items_t = int;
  using value_t     = int;
  using offset_t    = cub::detail::choose_offset_t<num_items_t>;

  int current_device{};
  REQUIRE(cudaSuccess == cudaGetDevice(&current_device));

  int ptx_version{};
  REQUIRE(cudaSuccess == cub::PtxVersion(ptx_version, current_device));

  num_items_t num_items = 1;

  auto d_in  = thrust::make_constant_iterator(value_t{1});
  auto d_out = thrust::device_vector<value_t>(1);

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(d_in, d_out.begin(), num_items));
  REQUIRE(d_out[0] == value_t{0});
}

template <int BlockThreads>
struct scan_tuning : cub::detail::scan::tuning<scan_tuning<BlockThreads>>
{
  template <class /* InputValueT */, class /* OutputValueT */, class AccumT, class /* Offset */, class /* ScanOpT */>
  struct fn
  {
    struct Policy500 : cub::ChainedPolicy<500, Policy500, Policy500>
    {
      struct ScanPolicyT
      {
        static constexpr int BLOCK_THREADS                      = BlockThreads;
        static constexpr int ITEMS_PER_THREAD                   = 1;
        static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = cub::BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE;

        static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::CacheLoadModifier::LOAD_DEFAULT;
        static constexpr cub::BlockStoreAlgorithm STORE_ALGORITHM =
          cub::BlockStoreAlgorithm::BLOCK_STORE_WARP_TRANSPOSE;
        static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING;

        struct detail
        {
          using delay_constructor_t = cub::detail::default_delay_constructor_t<AccumT>;
        };
      };
    };

    using MaxPolicy = Policy500;
  };
};

struct get_reduce_tuning_query_t
{};

struct reduce_tuning
{
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto query(const get_reduce_tuning_query_t&) const noexcept
  {
    return *this;
  }

  // Make sure this is not used
  template <class /* InputValueT */,
            class /* OutputValueT */,
            class /* AccumT */,
            class /* Offset */,
            class /* ScanOpT */>
  struct fn
  {};
};

using block_sizes = c2h::type_list<cuda::std::integral_constant<int, 32>, cuda::std::integral_constant<int, 64>>;

C2H_TEST("Device scan exclusive-scan can be tuned", "[scan][device]", block_sizes)
{
  constexpr int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_block_size(1);
  block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  auto num_items = 3;
  auto d_in      = thrust::make_constant_iterator(1);
  auto d_out     = thrust::device_vector<int>(num_items);

  // We are expecting that `reduce_tuning` is ignored
  auto env = cuda::execution::__tune(scan_tuning<target_block_size>{}, reduce_tuning{});

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveScan(d_in, d_out.begin(), block_size_check, 0, num_items, env));

  for (int i = 0; i < num_items; i++)
  {
    REQUIRE(d_out[i] == i);
  }
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Device scan exclusive-sum can be tuned", "[scan][device]", block_sizes)
{
  constexpr int target_block_size = c2h::get<0, TestType>::value;

  auto num_items = target_block_size;
  auto d_in      = thrust::make_constant_iterator(1);
  auto d_out     = thrust::device_vector<int>(num_items);

  // We are expecting that `reduce_tuning` is ignored
  auto env = cuda::execution::__tune(scan_tuning<target_block_size>{}, reduce_tuning{});

  REQUIRE(cudaSuccess == cub::DeviceScan::ExclusiveSum(d_in, d_out.begin(), num_items, env));

  for (int i = 0; i < num_items; i++)
  {
    REQUIRE(d_out[i] == i);
  }
}

#endif

using requirements = c2h::type_list<cuda::execution::determinism::run_to_run_t>;

C2H_TEST("Device scan exclusive-scan uses environment", "[scan][device]", requirements)
{
  using determinism_t = c2h::get<0, TestType>;

  using scan_op_t   = cuda::std::plus<>;
  using num_items_t = int;
  using offset_t    = cub::detail::choose_offset_t<num_items_t>;

  num_items_t num_items = 10;
  auto d_in             = thrust::make_constant_iterator(1.0f);
  auto d_out            = thrust::device_vector<float>(num_items);

  using input_it_t  = decltype(d_in);
  using output_it_t = decltype(d_out.begin());

  using init_t        = cub::detail::it_value_t<input_it_t>;
  using input_value_t = cub::detail::InputValue<init_t>;

  using accum_t =
    ::cuda::std::__accumulator_t<scan_op_t,
                                 cub::detail::it_value_t<input_it_t>,
                                 ::cuda::std::_If<::cuda::std::is_same_v<input_value_t, cub::NullType>,
                                                  cub::detail::it_value_t<input_it_t>,
                                                  typename input_value_t::value_type>>;

  init_t init{};
  size_t expected_bytes_allocated{};

  // To check if a given algorithm implementation is used, we check if associated kernels are invoked.
  auto kernels = [&]() {
    if constexpr (cuda::std::is_same_v<determinism_t, cuda::execution::determinism::run_to_run_t>)
    {
      REQUIRE(cudaSuccess
              == cub::DeviceScan::ExclusiveScan(
                nullptr, expected_bytes_allocated, d_in, d_out.begin(), scan_op_t{}, init, num_items));

      using policy_t =
        cub::detail::scan::policy_hub<cub::detail::it_value_t<input_it_t>,
                                      cub::detail::it_value_t<output_it_t>,
                                      accum_t,
                                      offset_t,
                                      scan_op_t>::MaxPolicy;

      using scan_tile_state_t = typename cub::ScanTileState<accum_t>;

      auto kernel1 = reinterpret_cast<void*>(
        cub::detail::scan::DeviceScanKernel<
          policy_t,
          input_it_t,
          output_it_t,
          scan_tile_state_t,
          scan_op_t,
          cub::detail::InputValue<init_t>,
          offset_t,
          accum_t,
          false,
          input_value_t::value_type>);

      auto kernel2 = reinterpret_cast<void*>(cub::detail::scan::DeviceScanInitKernel<scan_tile_state_t>);

      return cuda::std::array<void*, 2>{kernel1, kernel2};
    }
  }();

  // Equivalent to `cuexec::require(cuexec::determinism::run_to_run)` and
  //               `cuexec::require(cuexec::determinism::not_guaranteed)`
  auto env = stdexec::env{cuda::execution::require(determinism_t{}), // determinism
                          allowed_kernels(kernels), // allowed kernels for the given determinism
                          expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_scan_exclusive(d_in, d_out.begin(), scan_op_t{}, init, num_items, env);

  for (int i = 0; i < num_items; i++)
  {
    REQUIRE(d_out[i] == i);
  }
}

C2H_TEST("Device scan exclusive-sum uses environment", "[scan][device]", requirements)
{
  using determinism_t = c2h::get<0, TestType>;

  using scan_op_t   = cuda::std::plus<>;
  using num_items_t = int;
  using offset_t    = cub::detail::choose_offset_t<num_items_t>;

  num_items_t num_items = 10;
  auto d_in             = thrust::make_constant_iterator(1.0f);
  auto d_out            = thrust::device_vector<float>(num_items);

  using input_it_t  = decltype(d_in);
  using output_it_t = decltype(d_out.begin());

  using init_t        = cub::detail::it_value_t<input_it_t>;
  using input_value_t = cub::detail::InputValue<init_t>;

  using accum_t =
    ::cuda::std::__accumulator_t<scan_op_t,
                                 cub::detail::it_value_t<input_it_t>,
                                 ::cuda::std::_If<::cuda::std::is_same_v<input_value_t, cub::NullType>,
                                                  cub::detail::it_value_t<input_it_t>,
                                                  typename input_value_t::value_type>>;

  size_t expected_bytes_allocated{};

  // To check if a given algorithm implementation is used, we check if associated kernels are invoked.
  auto kernels = [&]() {
    if constexpr (cuda::std::is_same_v<determinism_t, cuda::execution::determinism::run_to_run_t>)
    {
      REQUIRE(cudaSuccess
              == cub::DeviceScan::ExclusiveSum(nullptr, expected_bytes_allocated, d_in, d_out.begin(), num_items));

      using policy_t =
        cub::detail::scan::policy_hub<cub::detail::it_value_t<input_it_t>,
                                      cub::detail::it_value_t<output_it_t>,
                                      accum_t,
                                      offset_t,
                                      scan_op_t>::MaxPolicy;

      using scan_tile_state_t = typename cub::ScanTileState<accum_t>;

      auto kernel1 = reinterpret_cast<void*>(
        cub::detail::scan::DeviceScanKernel<
          policy_t,
          input_it_t,
          output_it_t,
          scan_tile_state_t,
          scan_op_t,
          cub::detail::InputValue<init_t>,
          offset_t,
          accum_t,
          false,
          input_value_t::value_type>);

      auto kernel2 = reinterpret_cast<void*>(cub::detail::scan::DeviceScanInitKernel<scan_tile_state_t>);

      return cuda::std::array<void*, 2>{kernel1, kernel2};
    }
  }();

  // Equivalent to `cuexec::require(cuexec::determinism::run_to_run)` and
  //               `cuexec::require(cuexec::determinism::not_guaranteed)`
  auto env = stdexec::env{cuda::execution::require(determinism_t{}), // determinism
                          allowed_kernels(kernels), // allowed kernels for the given determinism
                          expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_scan_exclusive_sum(d_in, d_out.begin(), num_items, env);

  for (int i = 0; i < num_items; i++)
  {
    REQUIRE(d_out[i] == i);
  }
}
