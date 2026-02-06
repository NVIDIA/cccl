// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>

#include <cuda/iterator>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Reduce, device_reduce);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Sum, device_reduce_sum);

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
    *ptr = ActivePolicyT::SingleTilePolicy::BLOCK_THREADS;
    return cudaSuccess;
  }
};

TEST_CASE("Device reduce works with default environment", "[reduce][device]")
{
  using num_items_t = int;
  using value_t     = int;
  using offset_t    = cub::detail::choose_offset_t<num_items_t>;

  int current_device{};
  REQUIRE(cudaSuccess == cudaGetDevice(&current_device));

  cuda::arch_id arch_id{};
  REQUIRE(cudaSuccess == cub::detail::ptx_arch_id(arch_id, current_device));

  int target_block_size =
    cub::detail::reduce::policy_selector_from_types<value_t, offset_t, block_size_check_t>{}(arch_id)
      .single_tile.block_threads;

  num_items_t num_items = 1;
  c2h::device_vector<int> d_block_size(1);
  block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};
  auto d_in  = cuda::constant_iterator(value_t{1});
  auto d_out = thrust::device_vector<value_t>(1);

  REQUIRE(cudaSuccess == cub::DeviceReduce::Reduce(d_in, d_out.begin(), num_items, block_size_check, value_t{0}));
  REQUIRE(d_out[0] == num_items);

  // Make sure we use default tuning
  REQUIRE(d_block_size[0] == target_block_size);
}

TEST_CASE("Device sum works with default environment", "[reduce][device]")
{
  using num_items_t = int;
  using value_t     = int;
  using offset_t    = cub::detail::choose_offset_t<num_items_t>;

  int current_device{};
  REQUIRE(cudaSuccess == cudaGetDevice(&current_device));

  int ptx_version{};
  REQUIRE(cudaSuccess == cub::PtxVersion(ptx_version, current_device));

  num_items_t num_items = 1;

  auto d_in  = cuda::constant_iterator(value_t{1});
  auto d_out = thrust::device_vector<value_t>(1);

  REQUIRE(cudaSuccess == cub::DeviceReduce::Sum(d_in, d_out.begin(), num_items));
  REQUIRE(d_out[0] == num_items);
}

template <int BlockThreads>
struct reduce_tuning : cub::detail::reduce::tuning<reduce_tuning<BlockThreads>>
{
  _CCCL_API constexpr auto operator()(cuda::arch_id /*arch*/) const -> cub::detail::reduce::reduce_policy
  {
    const auto policy = cub::detail::reduce::agent_reduce_policy{
      BlockThreads, 1, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_DEFAULT};
    return {policy, policy, policy};
  }
};

struct get_scan_tuning_query_t
{};

struct scan_tuning
{
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto query(const get_scan_tuning_query_t&) const noexcept
  {
    return *this;
  }

  // Make sure this is not used
  template <class /* AccumT */, class /* Offset */, class /* OpT */>
  struct fn
  {};
};

using block_sizes = c2h::type_list<cuda::std::integral_constant<int, 32>, cuda::std::integral_constant<int, 64>>;

C2H_TEST("Device reduce can be tuned", "[reduce][device]", block_sizes)
{
  constexpr int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_block_size(1);
  block_size_check_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  auto num_items = 1;
  auto d_in      = cuda::constant_iterator(1);
  auto d_out     = thrust::device_vector<int>(1);

  // We are expecting that `scan_tuning` is ignored
  auto env = cuda::execution::__tune(reduce_tuning<target_block_size>{}, scan_tuning{});

  REQUIRE(cudaSuccess == cub::DeviceReduce::Reduce(d_in, d_out.begin(), num_items, block_size_check, 0, env));
  REQUIRE(d_out[0] == num_items);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Device sum can be tuned", "[reduce][device]", block_sizes)
{
  constexpr int target_block_size = c2h::get<0, TestType>::value;

  auto num_items = 1;
  auto d_in      = cuda::constant_iterator(1);
  auto d_out     = thrust::device_vector<int>(1);

  // We are expecting that `scan_tuning` is ignored
  auto env = cuda::execution::__tune(reduce_tuning<target_block_size>{}, scan_tuning{});

  REQUIRE(cudaSuccess == cub::DeviceReduce::Sum(d_in, d_out.begin(), num_items, env));
  REQUIRE(d_out[0] == num_items);
}
#endif

using requirements =
  c2h::type_list<cuda::execution::determinism::gpu_to_gpu_t,
                 cuda::execution::determinism::run_to_run_t,
                 cuda::execution::determinism::not_guaranteed_t>;

C2H_TEST("Device reduce uses environment", "[reduce][device]", requirements)
{
  using determinism_t = c2h::get<0, TestType>;
  using accumulator_t = float;
  using op_t          = cuda::std::plus<>;
  using num_items_t   = int;
  using offset_t      = cub::detail::choose_offset_t<num_items_t>;
  using transform_t   = cuda::std::identity;
  using init_t        = accumulator_t;

  num_items_t num_items = GENERATE(1 << 4, 1 << 24);
  auto d_in             = cuda::constant_iterator(1.0f);
  auto d_out            = thrust::device_vector<accumulator_t>(1);

  init_t init = 0;
  size_t expected_bytes_allocated{};

  // To check if a given algorithm implementation is used, we check if associated kernels are invoked.
  auto kernels = [&]() {
    if constexpr (std::is_same_v<determinism_t, cuda::execution::determinism::run_to_run_t>)
    {
      REQUIRE(
        cudaSuccess
        == cub::DeviceReduce::Reduce(nullptr, expected_bytes_allocated, d_in, d_out.begin(), num_items, op_t{}, init));

      using policy_t = cub::detail::reduce::policy_selector_from_types<accumulator_t, offset_t, op_t>;
      return cuda::std::array<void*, 3>{
        reinterpret_cast<void*>(
          cub::detail::reduce::DeviceReduceSingleTileKernel<
            policy_t,
            decltype(d_in),
            decltype(d_out.begin()),
            offset_t,
            op_t,
            init_t,
            accumulator_t,
            transform_t>),
        reinterpret_cast<void*>(
          cub::detail::reduce::DeviceReduceKernel<policy_t, decltype(d_in), offset_t, op_t, accumulator_t, transform_t>),
        reinterpret_cast<void*>(
          cub::detail::reduce::DeviceReduceSingleTileKernel<
            policy_t,
            accumulator_t*,
            decltype(d_out.begin()),
            int, // always used with int offset
            op_t,
            init_t,
            accumulator_t>)};
    }
    else if constexpr (cub::detail::is_non_deterministic_v<determinism_t>)
    {
      using policy_t = cub::detail::reduce::policy_selector_from_types<accumulator_t, offset_t, op_t>;
      auto* raw_ptr  = thrust::raw_pointer_cast(d_out.data());

      REQUIRE(
        cudaSuccess
        == cub::detail::reduce::dispatch_nondeterministic(
          nullptr, expected_bytes_allocated, d_in, raw_ptr, num_items, op_t{}, init, /* stream */ 0, transform_t{}));

      return cuda::std::array<void*, 1>{reinterpret_cast<void*>(
        cub::detail::reduce::NondeterministicDeviceReduceAtomicKernel<
          policy_t,
          decltype(d_in),
          decltype(raw_ptr),
          offset_t,
          op_t,
          init_t,
          accumulator_t,
          transform_t>)};
    }
    else
    {
      using policy_t              = cub::detail::rfa::policy_hub<accumulator_t, offset_t, op_t>::MaxPolicy;
      using deterministic_add_t   = cub::detail::rfa::deterministic_sum_t<accumulator_t>;
      using reduction_op_t        = deterministic_add_t;
      using deterministic_accum_t = deterministic_add_t::DeterministicAcc;
      using output_it_t           = decltype(d_out.begin());

      using dispatch_t = cub::detail::rfa::
        dispatch_t<decltype(d_in), decltype(d_out.begin()), offset_t, init_t, transform_t, accumulator_t>;

      REQUIRE(
        cudaSuccess == dispatch_t::Dispatch(nullptr, expected_bytes_allocated, d_in, d_out.begin(), num_items, init));

      auto k1 = cub::detail::reduce::DeterministicDeviceReduceSingleTileKernel<
        policy_t,
        decltype(d_in),
        output_it_t,
        reduction_op_t,
        init_t,
        deterministic_accum_t,
        transform_t>;
      auto k2 = cub::detail::reduce::
        DeterministicDeviceReduceKernel<policy_t, decltype(d_in), reduction_op_t, deterministic_accum_t, transform_t>;
      auto k3 = cub::detail::reduce::DeterministicDeviceReduceSingleTileKernel<
        policy_t,
        deterministic_accum_t*,
        output_it_t,
        reduction_op_t,
        init_t,
        deterministic_accum_t,
        transform_t>;
      // TODO(bgruber): enable this when we have Catch2 3.13+
      // UNSCOPED_CAPTURE(c2h::type_name<decltype(k1)>(), c2h::type_name<decltype(k2)>(),
      // c2h::type_name<decltype(k3)>());
      return cuda::std::array<void*, 3>{
        reinterpret_cast<void*>(k1), reinterpret_cast<void*>(k2), reinterpret_cast<void*>(k3)};
    }
  }();

  // Equivalent to `cuexec::require(cuexec::determinism::run_to_run)` and
  //               `cuexec::require(cuexec::determinism::not_guaranteed)`
  auto env = stdexec::env{cuda::execution::require(determinism_t{}), // determinism
                          allowed_kernels(kernels), // allowed kernels for the given determinism
                          expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_reduce(d_in, d_out.begin(), num_items, op_t{}, init, env);

  REQUIRE(d_out[0] == num_items);
}

C2H_TEST("Device sum uses environment", "[reduce][device]", requirements)
{
  using determinism_t = c2h::get<0, TestType>;
  using accumulator_t = float;
  using op_t          = cuda::std::plus<>;
  using num_items_t   = int;
  using offset_t      = cub::detail::choose_offset_t<num_items_t>;
  using transform_t   = cuda::std::identity;
  using init_t        = accumulator_t;

  num_items_t num_items = GENERATE(1 << 4, 1 << 24);
  auto d_in             = cuda::constant_iterator(1.0f);
  auto d_out            = thrust::device_vector<accumulator_t>(1);

  [[maybe_unused]] init_t init = 0;
  size_t expected_bytes_allocated{};

  // To check if a given algorithm implementation is used, we check if associated kernels are invoked.
  auto kernels = [&]() {
    if constexpr (std::is_same_v<determinism_t, cuda::execution::determinism::run_to_run_t>)
    {
      REQUIRE(cudaSuccess == cub::DeviceReduce::Sum(nullptr, expected_bytes_allocated, d_in, d_out.begin(), num_items));

      using policy_t = cub::detail::reduce::policy_selector_from_types<accumulator_t, offset_t, op_t>;
      return cuda::std::array<void*, 3>{
        reinterpret_cast<void*>(
          cub::detail::reduce::DeviceReduceSingleTileKernel<
            policy_t,
            decltype(d_in),
            decltype(d_out.begin()),
            offset_t,
            op_t,
            init_t,
            accumulator_t,
            transform_t>),
        reinterpret_cast<void*>(
          cub::detail::reduce::DeviceReduceKernel<policy_t, decltype(d_in), offset_t, op_t, accumulator_t, transform_t>),
        reinterpret_cast<void*>(
          cub::detail::reduce::DeviceReduceSingleTileKernel<
            policy_t,
            accumulator_t*,
            decltype(d_out.begin()),
            int, // always used with int offset
            op_t,
            init_t,
            accumulator_t>)};
    }
    else if constexpr (cub::detail::is_non_deterministic_v<determinism_t>)
    {
      using policy_t = cub::detail::reduce::policy_selector_from_types<accumulator_t, offset_t, op_t>;
      auto* raw_ptr  = thrust::raw_pointer_cast(d_out.data());

      REQUIRE(
        cudaSuccess
        == cub::detail::reduce::dispatch_nondeterministic(
          nullptr, expected_bytes_allocated, d_in, raw_ptr, num_items, op_t{}, init, /* stream */ 0, transform_t{}));

      return cuda::std::array<void*, 1>{reinterpret_cast<void*>(
        cub::detail::reduce::NondeterministicDeviceReduceAtomicKernel<
          policy_t,
          decltype(d_in),
          decltype(raw_ptr),
          offset_t,
          op_t,
          init_t,
          accumulator_t,
          transform_t>)};
    }
    else
    {
      using policy_t              = cub::detail::rfa::policy_hub<accumulator_t, offset_t, op_t>::MaxPolicy;
      using deterministic_add_t   = cub::detail::rfa::deterministic_sum_t<accumulator_t>;
      using reduction_op_t        = deterministic_add_t;
      using deterministic_accum_t = deterministic_add_t::DeterministicAcc;
      using output_it_t           = decltype(d_out.begin());

      using dispatch_t = cub::detail::rfa::
        dispatch_t<decltype(d_in), decltype(d_out.begin()), offset_t, init_t, transform_t, accumulator_t>;

      REQUIRE(
        cudaSuccess == dispatch_t::Dispatch(nullptr, expected_bytes_allocated, d_in, d_out.begin(), num_items, init));

      auto k1 = cub::detail::reduce::DeterministicDeviceReduceSingleTileKernel<
        policy_t,
        decltype(d_in),
        output_it_t,
        reduction_op_t,
        init_t,
        deterministic_accum_t,
        transform_t>;
      auto k2 = cub::detail::reduce::
        DeterministicDeviceReduceKernel<policy_t, decltype(d_in), reduction_op_t, deterministic_accum_t, transform_t>;
      auto k3 = cub::detail::reduce::DeterministicDeviceReduceSingleTileKernel<
        policy_t,
        deterministic_accum_t*,
        output_it_t,
        reduction_op_t,
        init_t,
        deterministic_accum_t,
        transform_t>;
      // TODO(bgruber): enable this when we have Catch2 3.13+
      // UNSCOPED_CAPTURE(c2h::type_name<decltype(k1)>(), c2h::type_name<decltype(k2)>(),
      // c2h::type_name<decltype(k3)>());
      return cuda::std::array<void*, 3>{
        reinterpret_cast<void*>(k1), reinterpret_cast<void*>(k2), reinterpret_cast<void*>(k3)};
    }
  }();

  // Equivalent to `cuexec::require(cuexec::determinism::run_to_run)` and
  //               `cuexec::require(cuexec::determinism::not_guaranteed)`
  auto env = stdexec::env{cuda::execution::require(determinism_t{}), // determinism
                          allowed_kernels(kernels), // allowed kernels for the given determinism
                          expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_reduce_sum(d_in, d_out.begin(), num_items, env);

  REQUIRE(d_out[0] == num_items);
}
