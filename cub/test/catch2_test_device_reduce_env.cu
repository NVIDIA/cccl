// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/iterator>
#include <cuda/stream>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Reduce, device_reduce);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Sum, device_reduce_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Min, device_reduce_min);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Max, device_reduce_max);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::TransformReduce, device_transform_reduce);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ReduceByKey, device_reduce_by_key);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMin, device_arg_min);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMax, device_arg_max);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

template <int ThreadsPerBlock>
struct reduce_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::detail::reduce::reduce_policy
  {
    const auto policy = cub::detail::reduce::agent_reduce_policy{
      ThreadsPerBlock, 1, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_DEFAULT};
    return {policy, policy};
  }
};

template <int ThreadsPerBlock>
struct nondeterministic_reduce_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const
    -> cub::detail::reduce_nondeterministic::reduce_nondeterministic_policy
  {
    return {cub::detail::reduce::agent_reduce_policy{
      ThreadsPerBlock, 1, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC, cub::LOAD_DEFAULT}};
  }
};

template <int ThreadsPerBlock>
struct deterministic_reduce_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::detail::rfa::rfa_policy
  {
    return {cub::detail::rfa::reduce_policy{ThreadsPerBlock, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS},
            cub::detail::rfa::single_tile_policy{ThreadsPerBlock, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS}};
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

using block_size_check_plus_t = block_size_extracting_op<cuda::std::plus<>>;

// Launcher helper always passes an environment.
// We need a test of simple use to check if default environment works.
// ifdef it out not to spend time compiling and running it twice.
#if TEST_LAUNCH == 0
TEST_CASE("Device reduce works with default environment", "[reduce][device]")
{
  using num_items_t = int;
  using value_t     = int;
  using offset_t    = cub::detail::choose_offset_t<num_items_t>;

  int current_device{};
  REQUIRE(cudaSuccess == cudaGetDevice(&current_device));

  cuda::compute_capability cc{};
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc, current_device));

  unsigned int target_block_size =
    cub::detail::reduce::policy_selector_from_types<value_t, offset_t, block_size_check_plus_t>{}(cc)
      .single_tile.threads_per_block;

  num_items_t num_items = 1;
  c2h::device_vector<unsigned int> d_block_size(1);
  block_size_check_plus_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};
  auto d_in  = cuda::constant_iterator(value_t{1});
  auto d_out = thrust::device_vector<value_t>(1);

  REQUIRE(cudaSuccess == cub::DeviceReduce::Reduce(d_in, d_out.begin(), num_items, block_size_check, value_t{0}));
  REQUIRE(d_out[0] == num_items);

  // Make sure we use default tuning
  REQUIRE(d_block_size[0] == target_block_size);
}

TEST_CASE("Device Sum works with default environment", "[reduce][device]")
{
  using num_items_t = int;
  using value_t     = int;

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
#endif

#if TEST_LAUNCH != 1
C2H_TEST("Device reduce can be tuned", "[reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);

  auto d_in  = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto d_out = thrust::device_vector<int>(1);

  auto env = cuda::execution::tune(
    reduce_tuning<target_block_size>{}, // <-- should be taken
#  if _CCCL_CUDA_COMPILER(NVCC, >=, 12, 1)
    // some rare combinations, like nvcc 12.0 + clang-14 in C++20, or nvcc 12.0 + GCC12 fail with:
    //   pod_tuple.h(130): error: Internal Compiler Error (codegen): "internal error during structure layout!"
    // if we pass more than two policy selectors
    unrelated_tuning{}, // should be ignored
    nondeterministic_reduce_tuning<target_block_size * 2>{}, // should be ignored
#  endif // _CCCL_CUDA_COMPILER(NVCC, >=, 12, 1)
    deterministic_reduce_tuning<target_block_size * 2>{} // should be ignored
  );

  device_reduce(d_in, d_out.begin(), 1, cuda::std::plus<>{}, 0, env);
  REQUIRE(d_out[0] == 1);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Device reduce not_guaranteed can be tuned", "[reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);

  auto d_in  = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto d_out = thrust::device_vector<int>(1);

  auto env = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::not_guaranteed),
    cuda::execution::tune(
#  if _CCCL_CUDA_COMPILER(NVCC, >=, 12, 1)
      reduce_tuning<target_block_size * 2>{}, // should be ignored
      unrelated_tuning{}, // should be ignored
#  endif // _CCCL_CUDA_COMPILER(NVCC, >=, 12, 1)
      nondeterministic_reduce_tuning<target_block_size>{}, // <-- should be taken
      deterministic_reduce_tuning<target_block_size * 2>{} // should be ignored
      )};

  device_reduce(d_in, d_out.begin(), 1, cuda::std::plus<>{}, 0, env);
  REQUIRE(d_out[0] == 1);
  REQUIRE(d_block_size[0] == target_block_size);
}
C2H_TEST("Device reduce run_to_run can be tuned", "[reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);

  auto d_in  = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto d_out = thrust::device_vector<int>(1);

  auto env = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::run_to_run),
    cuda::execution::tune(
      reduce_tuning<target_block_size>{}, // <-- should be taken
#  if _CCCL_CUDA_COMPILER(NVCC, >=, 12, 1)
      unrelated_tuning{}, // should be ignored
      nondeterministic_reduce_tuning<target_block_size * 2>{}, // should be ignored
#  endif // _CCCL_CUDA_COMPILER(NVCC, >=, 12, 1)
      deterministic_reduce_tuning<target_block_size * 2>{} // should be ignored
      )};

  device_reduce(d_in, d_out.begin(), 1, cuda::std::plus<>{}, 0, env);
  REQUIRE(d_out[0] == 1);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Device reduce gpu_to_gpu can be tuned", "[reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);

  auto d_in  = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto d_out = thrust::device_vector<int>(1);

  auto env = cuda::std::execution::env{
    cuda::execution::require(cuda::execution::determinism::gpu_to_gpu),
    cuda::execution::tune(
#  if _CCCL_CUDA_COMPILER(NVCC, >=, 12, 1)
      reduce_tuning<target_block_size * 2>{}, // should be ignored
      unrelated_tuning{}, // should be ignored
#  endif // _CCCL_CUDA_COMPILER(NVCC, >=, 12, 1)
      nondeterministic_reduce_tuning<target_block_size * 2>{}, // // should be ignored
      deterministic_reduce_tuning<target_block_size>{} // <-- should be taken
      )};

  device_reduce(d_in, d_out.begin(), 1, cuda::std::plus<float>{}, 0, env); // make accum_t float to select RFA
  REQUIRE(d_out[0] == 1);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Device Sum can be tuned", "[reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);

  auto d_in  = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto d_out = c2h::device_vector<int>(1);

  auto env = cuda::execution::tune(reduce_tuning<target_block_size>{});

  device_reduce_sum(d_in, d_out.begin(), 1, env);
  REQUIRE(d_out[0] == 1);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Device Min can be tuned", "[reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);

  auto d_in  = block_size_extracting_constant_iterator(42, thrust::raw_pointer_cast(d_block_size.data()));
  auto d_out = c2h::device_vector<int>(1);

  auto env = cuda::execution::tune(reduce_tuning<target_block_size>{});

  device_reduce_min(d_in, d_out.begin(), 1, env);
  REQUIRE(d_out[0] == 42);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Device Max can be tuned", "[reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);

  auto d_in  = block_size_extracting_constant_iterator(42, thrust::raw_pointer_cast(d_block_size.data()));
  auto d_out = c2h::device_vector<int>(1);

  auto env = cuda::execution::tune(reduce_tuning<target_block_size>{});

  device_reduce_max(d_in, d_out.begin(), 1, env);
  REQUIRE(d_out[0] == 42);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Device TransformReduce can be tuned", "[reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);

  auto d_in  = block_size_extracting_constant_iterator(1, thrust::raw_pointer_cast(d_block_size.data()));
  auto d_out = c2h::device_vector<int>(1);

  auto env = cuda::execution::tune(
    reduce_tuning<target_block_size>{},
#  if _CCCL_CUDA_COMPILER(NVCC, >=, 12, 1)
    unrelated_tuning{}, // should be ignored
#  endif // _CCCL_CUDA_COMPILER(NVCC, >=, 12, 9)
    nondeterministic_reduce_tuning<target_block_size * 2>{} // should be ignored
  );

  device_transform_reduce(d_in, d_out.begin(), 1, cuda::std::plus<>{}, cuda::std::negate<int>{}, 0, env);
  REQUIRE(d_out[0] == -1);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Device ArgMin can be tuned", "[reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);
  using compare_t = block_size_extracting_op<cuda::std::less<>>;
  compare_t compare_op{thrust::raw_pointer_cast(d_block_size.data())};

  auto input        = c2h::device_vector<int>{3, 1, 4, 0, 2};
  auto min_output   = c2h::device_vector<int>(1);
  auto index_output = c2h::device_vector<cuda::std::int64_t>(1);

  auto env = cuda::execution::tune(reduce_tuning<target_block_size>{});

  device_arg_min(
    input.begin(), min_output.begin(), index_output.begin(), static_cast<int>(input.size()), compare_op, env);
  REQUIRE(min_output[0] == 0);
  REQUIRE(index_output[0] == 3);
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("Device ArgMax can be tuned", "[reduce][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);
  using compare_t = block_size_extracting_op<cuda::std::less<>>;
  compare_t compare_op{thrust::raw_pointer_cast(d_block_size.data())};

  auto input        = c2h::device_vector<int>{3, 1, 4, 0, 2};
  auto max_output   = c2h::device_vector<int>(1);
  auto index_output = c2h::device_vector<cuda::std::int64_t>(1);

  auto env = cuda::execution::tune(reduce_tuning<target_block_size>{});

  device_arg_max(
    input.begin(), max_output.begin(), index_output.begin(), static_cast<int>(input.size()), compare_op, env);
  REQUIRE(max_output[0] == 4);
  REQUIRE(index_output[0] == 2);
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH != 1

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
      using policy_t = cub::detail::reduce_nondeterministic::policy_selector_from_types<accumulator_t, offset_t, op_t>;
      auto* raw_ptr  = thrust::raw_pointer_cast(d_out.data());

      REQUIRE(
        cudaSuccess
        == cub::detail::reduce_nondeterministic::dispatch(
          nullptr,
          expected_bytes_allocated,
          d_in,
          raw_ptr,
          num_items,
          op_t{},
          init,
          /* stream */ nullptr,
          transform_t{}));

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
      using policy_t              = cub::detail::rfa::policy_selector_from_types<accumulator_t>;
      using deterministic_add_t   = cub::detail::rfa::deterministic_sum_t<accumulator_t>;
      using reduction_op_t        = deterministic_add_t;
      using deterministic_accum_t = deterministic_add_t::DeterministicAcc;
      using output_it_t           = decltype(d_out.begin());

      REQUIRE(cudaSuccess
              == cub::detail::rfa::
                dispatch<decltype(d_in), decltype(d_out.begin()), offset_t, init_t, transform_t, accumulator_t>(
                  nullptr, expected_bytes_allocated, d_in, d_out.begin(), num_items, init));

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
      using policy_t = cub::detail::reduce_nondeterministic::policy_selector_from_types<accumulator_t, offset_t, op_t>;
      auto* raw_ptr  = thrust::raw_pointer_cast(d_out.data());

      REQUIRE(
        cudaSuccess
        == cub::detail::reduce_nondeterministic::dispatch(
          nullptr,
          expected_bytes_allocated,
          d_in,
          raw_ptr,
          num_items,
          op_t{},
          init,
          /* stream */ nullptr,
          transform_t{}));

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
      using policy_t              = cub::detail::rfa::policy_selector_from_types<accumulator_t>;
      using deterministic_add_t   = cub::detail::rfa::deterministic_sum_t<accumulator_t>;
      using reduction_op_t        = deterministic_add_t;
      using deterministic_accum_t = deterministic_add_t::DeterministicAcc;
      using output_it_t           = decltype(d_out.begin());

      REQUIRE(cudaSuccess
              == cub::detail::rfa::
                dispatch<decltype(d_in), decltype(d_out.begin()), offset_t, init_t, transform_t, accumulator_t>(
                  nullptr, expected_bytes_allocated, d_in, d_out.begin(), num_items, init));

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

#if TEST_LAUNCH == 0

TEST_CASE("Device Min works with default environment", "[reduce][device]")
{
  auto input  = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto output = c2h::device_vector<float>(1);

  auto error = cub::DeviceReduce::Min(input.begin(), output.begin(), static_cast<int>(input.size()));
  REQUIRE(error == cudaSuccess);
  REQUIRE(output[0] == 0.0f);
}

TEST_CASE("Device Max works with default environment", "[reduce][device]")
{
  auto input  = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto output = c2h::device_vector<float>(1);

  auto error = cub::DeviceReduce::Max(input.begin(), output.begin(), static_cast<int>(input.size()));
  REQUIRE(error == cudaSuccess);
  REQUIRE(output[0] == 4.0f);
}

TEST_CASE("Device TransformReduce works with default environment", "[reduce][device]")
{
  auto d_in  = c2h::device_vector<int>{1, 2, 3, 4};
  auto d_out = thrust::device_vector<int>(1);

  auto negate = cuda::std::negate<int>{};

  REQUIRE(cudaSuccess
          == cub::DeviceReduce::TransformReduce(
            d_in.begin(), d_out.begin(), static_cast<int>(d_in.size()), cuda::std::plus<int>{}, negate, 0));

  REQUIRE(d_out[0] == -10);
}

TEST_CASE("Device ReduceByKey works with default environment", "[reduce][device]")
{
  auto d_keys_in        = c2h::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto d_values_in      = c2h::device_vector<int>{0, 7, 1, 6, 2, 5, 3, 4};
  auto d_unique_out     = c2h::device_vector<int>(8);
  auto d_aggregates_out = c2h::device_vector<int>(8);
  auto d_num_runs_out   = c2h::device_vector<int>(1);

  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::ReduceByKey(
      d_keys_in.begin(),
      d_unique_out.begin(),
      d_values_in.begin(),
      d_aggregates_out.begin(),
      d_num_runs_out.begin(),
      cuda::minimum<int>{},
      static_cast<int>(d_keys_in.size())));

  REQUIRE(d_num_runs_out[0] == 5);
  c2h::device_vector<int> expected_keys{0, 2, 9, 5, 8};
  c2h::device_vector<int> expected_aggregates{0, 1, 6, 2, 4};
  d_unique_out.resize(5);
  d_aggregates_out.resize(5);
  REQUIRE(d_unique_out == expected_keys);
  REQUIRE(d_aggregates_out == expected_aggregates);
}

TEST_CASE("Device ArgMin works with default environment", "[reduce][device]")
{
  auto input        = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_output   = c2h::device_vector<float>(1);
  auto index_output = c2h::device_vector<cuda::std::int64_t>(1);

  auto error =
    cub::DeviceReduce::ArgMin(input.begin(), min_output.begin(), index_output.begin(), static_cast<int>(input.size()));
  REQUIRE(error == cudaSuccess);
  REQUIRE(min_output[0] == 0.0f);
  REQUIRE(index_output[0] == 3);
}

TEST_CASE("Device ArgMax works with default environment", "[reduce][device]")
{
  auto input        = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto max_output   = c2h::device_vector<float>(1);
  auto index_output = c2h::device_vector<cuda::std::int64_t>(1);

  auto error =
    cub::DeviceReduce::ArgMax(input.begin(), max_output.begin(), index_output.begin(), static_cast<int>(input.size()));
  REQUIRE(error == cudaSuccess);
  REQUIRE(max_output[0] == 4.0f);
  REQUIRE(index_output[0] == 2);
}

TEST_CASE("Device ArgMin with compare_op works with default environment", "[reduce][device]")
{
  auto input        = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_output   = c2h::device_vector<float>(1);
  auto index_output = c2h::device_vector<cuda::std::int64_t>(1);

  auto error = cub::DeviceReduce::ArgMin(
    input.begin(), min_output.begin(), index_output.begin(), static_cast<int>(input.size()), cuda::std::less{});
  REQUIRE(error == cudaSuccess);
  REQUIRE(min_output[0] == 0.0f);
  REQUIRE(index_output[0] == 3);
}

TEST_CASE("Device ArgMax with compare_op works with default environment", "[reduce][device]")
{
  auto input        = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto max_output   = c2h::device_vector<float>(1);
  auto index_output = c2h::device_vector<cuda::std::int64_t>(1);

  auto error = cub::DeviceReduce::ArgMax(
    input.begin(), max_output.begin(), index_output.begin(), static_cast<int>(input.size()), cuda::std::less{});
  REQUIRE(error == cudaSuccess);
  REQUIRE(max_output[0] == 4.0f);
  REQUIRE(index_output[0] == 2);
}

#endif

C2H_TEST("Device TransformReduce uses environment", "[reduce][device]")
{
  auto d_in  = c2h::device_vector<int>{1, 2, 3, 4};
  auto d_out = thrust::device_vector<int>(1);

  auto negate = cuda::std::negate<int>{};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::TransformReduce(
      nullptr,
      expected_bytes_allocated,
      d_in.begin(),
      d_out.begin(),
      static_cast<int>(d_in.size()),
      cuda::std::plus<int>{},
      negate,
      0));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_transform_reduce(
    d_in.begin(), d_out.begin(), static_cast<int>(d_in.size()), cuda::std::plus<int>{}, negate, 0, env);

  REQUIRE(d_out[0] == -10);
}

C2H_TEST("Device Min uses environment", "[reduce][device]")
{
  auto input  = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto output = c2h::device_vector<float>(1);

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceReduce::Min(
            nullptr, expected_bytes_allocated, input.begin(), output.begin(), static_cast<int>(input.size())));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_reduce_min(input.begin(), output.begin(), static_cast<int>(input.size()), env);

  REQUIRE(output[0] == 0.0f);
}

C2H_TEST("Device Max uses environment", "[reduce][device]")
{
  auto input  = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto output = c2h::device_vector<float>(1);

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceReduce::Max(
            nullptr, expected_bytes_allocated, input.begin(), output.begin(), static_cast<int>(input.size())));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_reduce_max(input.begin(), output.begin(), static_cast<int>(input.size()), env);

  REQUIRE(output[0] == 4.0f);
}

C2H_TEST("Device ReduceByKey uses environment", "[reduce][device]")
{
  auto d_keys_in        = c2h::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto d_values_in      = c2h::device_vector<int>{0, 7, 1, 6, 2, 5, 3, 4};
  auto d_unique_out     = c2h::device_vector<int>(8);
  auto d_aggregates_out = c2h::device_vector<int>(8);
  auto d_num_runs_out   = c2h::device_vector<int>(1);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::ReduceByKey(
      nullptr,
      expected_bytes_allocated,
      d_keys_in.begin(),
      d_unique_out.begin(),
      d_values_in.begin(),
      d_aggregates_out.begin(),
      d_num_runs_out.begin(),
      cuda::minimum<int>{},
      static_cast<int>(d_keys_in.size())));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_reduce_by_key(
    d_keys_in.begin(),
    d_unique_out.begin(),
    d_values_in.begin(),
    d_aggregates_out.begin(),
    d_num_runs_out.begin(),
    cuda::minimum<int>{},
    static_cast<int>(d_keys_in.size()),
    env);

  REQUIRE(d_num_runs_out[0] == 5);
  c2h::device_vector<int> expected_keys{0, 2, 9, 5, 8};
  c2h::device_vector<int> expected_aggregates{0, 1, 6, 2, 4};
  d_unique_out.resize(5);
  d_aggregates_out.resize(5);
  REQUIRE(d_unique_out == expected_keys);
  REQUIRE(d_aggregates_out == expected_aggregates);
}

C2H_TEST("Device ArgMin uses environment", "[reduce][device]")
{
  auto input        = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_output   = c2h::device_vector<float>(1);
  auto index_output = c2h::device_vector<cuda::std::int64_t>(1);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::ArgMin(
      nullptr,
      expected_bytes_allocated,
      input.begin(),
      min_output.begin(),
      index_output.begin(),
      static_cast<int>(input.size())));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_arg_min(input.begin(), min_output.begin(), index_output.begin(), static_cast<int>(input.size()), env);

  REQUIRE(min_output[0] == 0.0f);
  REQUIRE(index_output[0] == 3);
}

C2H_TEST("Device ArgMin with compare_op uses environment", "[reduce][device]")
{
  auto input        = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_output   = c2h::device_vector<float>(1);
  auto index_output = c2h::device_vector<cuda::std::int64_t>(1);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::ArgMin(
      nullptr,
      expected_bytes_allocated,
      input.begin(),
      min_output.begin(),
      index_output.begin(),
      static_cast<int>(input.size()),
      cuda::std::less{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_arg_min(
    input.begin(), min_output.begin(), index_output.begin(), static_cast<int>(input.size()), cuda::std::less{}, env);

  REQUIRE(min_output[0] == 0.0f);
  REQUIRE(index_output[0] == 3);
}

C2H_TEST("Device ArgMax uses environment", "[reduce][device]")
{
  auto input        = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto max_output   = c2h::device_vector<float>(1);
  auto index_output = c2h::device_vector<cuda::std::int64_t>(1);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::ArgMax(
      nullptr,
      expected_bytes_allocated,
      input.begin(),
      max_output.begin(),
      index_output.begin(),
      static_cast<int>(input.size())));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_arg_max(input.begin(), max_output.begin(), index_output.begin(), static_cast<int>(input.size()), env);

  REQUIRE(max_output[0] == 4.0f);
  REQUIRE(index_output[0] == 2);
}

C2H_TEST("Device ArgMax with compare_op uses environment", "[reduce][device]")
{
  auto input        = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto max_output   = c2h::device_vector<float>(1);
  auto index_output = c2h::device_vector<cuda::std::int64_t>(1);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::ArgMax(
      nullptr,
      expected_bytes_allocated,
      input.begin(),
      max_output.begin(),
      index_output.begin(),
      static_cast<int>(input.size()),
      cuda::std::less{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_arg_max(
    input.begin(), max_output.begin(), index_output.begin(), static_cast<int>(input.size()), cuda::std::less{}, env);

  REQUIRE(max_output[0] == 4.0f);
  REQUIRE(index_output[0] == 2);
}
