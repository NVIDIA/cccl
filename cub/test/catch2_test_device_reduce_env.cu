// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/config.cuh>

#include <cub/device/device_reduce.cuh>

#include <thrust/functional.h>
#include <thrust/tabulate.h>

#include <cuda/__functional/call_or.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/devices>
#include <cuda/execution>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/concepts>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/numeric>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cstddef>
#include <sstream>

#include <cuda_runtime_api.h>

#include "block_size_extracting_helpers.h"
#include "catch2_test_custom_streams.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include "stream_registry_factory.h"
#include <c2h/buffer_utils.cuh>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/device_and_stream.h>
#include <c2h/device_policy.h>

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceReduce::Reduce, device_reduce);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceReduce::Sum, device_reduce_sum);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceReduce::Min, device_reduce_min);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceReduce::Max, device_reduce_max);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceReduce::TransformReduce, device_transform_reduce);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceReduce::ReduceByKey, device_reduce_by_key);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceReduce::ArgMin, device_arg_min);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceReduce::ArgMax, device_arg_max);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceReduce::ArgMinMax, device_arg_minmax);

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Reduce, device_reduce_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Sum, device_reduce_sum_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Min, device_reduce_min_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Max, device_reduce_max_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::TransformReduce, device_transform_reduce_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ReduceByKey, device_reduce_by_key_two_phase);

// %PARAM% TEST_LAUNCH lid 0:1:2

namespace stdexec = cuda::std::execution;

template <typename T>
void check_output_prefix(cuda::stream_ref stream,
                         cuda::device_ref device,
                         const cuda::device_buffer<T>& output,
                         const cuda::host_buffer<T>& expected)
{
  auto result = c2h::make_host_buffer<T>(stream, device, expected.size(), cuda::no_init);
  c2h::copy_to_host(stream, output.first(expected.size()), result);
  REQUIRE(cuda::std::equal(result.begin(), result.end(), expected.begin()));
}

using cuda::execution::determinism::__determinism_t;

template <int ThreadsPerBlock>
struct reduce_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::ReducePolicy
  {
    const auto policy =
      cub::ReducePassPolicy{ThreadsPerBlock, 1, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_DEFAULT};
    return {policy, policy};
  }
};

template <int ThreadsPerBlock>
struct reduce_tuning_transform
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability, cub::ReducePolicy policy) const
    -> cub::ReducePolicy
  {
    policy.multi_tile.threads_per_block  = ThreadsPerBlock;
    policy.single_tile.threads_per_block = ThreadsPerBlock;
    return policy;
  }
};

// Selects a non-deterministic block reduce. This is only valid if the reduction is not required to be run-to-run
// deterministic, which includes a determinism::not_guaranteed request that falls back to a run-to-run deterministic
// dispatch (e.g., for min/max or arg-min/arg-max).
struct nondeterministic_reduce_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::ReducePolicy
  {
    const auto policy =
      cub::ReducePassPolicy{128, 4, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC, cub::LOAD_DEFAULT};
    return {policy, policy};
  }
};

// not a known operator, so the reduction is not considered stable
struct custom_max_op
{
  __host__ __device__ int operator()(int a, int b) const
  {
    return a < b ? b : a;
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
CUB_TEST_CASE("Device reduce works with default environment", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  using num_items_t = int;
  using value_t     = int;
  using offset_t    = cub::detail::choose_offset_t<num_items_t>;

  cuda::compute_capability cc{};
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc));

  const unsigned int target_block_size =
    cub::detail::reduce::policy_selector_from_types<value_t, offset_t, block_size_check_plus_t>{}(cc)
      .single_tile.threads_per_block;

  const num_items_t num_items = 1;
  auto d_block_size           = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const block_size_check_plus_t block_size_check{d_block_size.data()};
  auto d_in  = cuda::constant_iterator(value_t{1});
  auto d_out = c2h::make_device_buffer<value_t>(stream, device, 1, cuda::no_init);

  REQUIRE(cudaSuccess == cub::DeviceReduce::Reduce(d_in, d_out.data(), num_items, block_size_check, value_t{0}));
  REQUIRE(c2h::read_single(stream, d_out) == num_items);

  // Make sure we use default tuning
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST_CASE("Device Sum works with default environment", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  using num_items_t           = int;
  using value_t               = int;
  const num_items_t num_items = 1;

  auto d_in  = cuda::constant_iterator(value_t{1});
  auto d_out = c2h::make_device_buffer<value_t>(stream, device, 1, cuda::no_init);

  REQUIRE(cudaSuccess == cub::DeviceReduce::Sum(d_in, d_out.data(), num_items));
  REQUIRE(c2h::read_single(stream, d_out) == num_items);
}
#endif

#if TEST_LAUNCH != 1
CUB_TEST("Device reduce can be tuned", "[reduce][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});

  auto d_in  = block_size_extracting_constant_iterator(1, d_block_size.data());
  auto d_out = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);

  auto env = cuda::execution::tune(reduce_tuning<target_block_size>{}, unrelated_tuning{});

  stream.sync();
  device_reduce(d_in, d_out.data(), 1, cuda::std::plus<>{}, 0, env);
  REQUIRE(c2h::read_single(stream, d_out) == 1);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST_CASE("Device reduce tuning can be modified", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = 64; // must be different from any value we use in production
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});

  auto d_in  = block_size_extracting_constant_iterator(1, d_block_size.data());
  auto d_out = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);

  auto env = cuda::execution::tune(reduce_tuning_transform<target_block_size>{});

  {
    stream.sync();
    device_reduce(d_in, d_out.data(), 1, cuda::std::plus<>{}, 0, stdexec::env<>{});
    const unsigned int default_block_size = c2h::read_single(stream, d_block_size);
    REQUIRE(default_block_size != target_block_size);
    cuda::fill_bytes(stream, d_block_size, 0);
  }

  stream.sync();
  device_reduce(d_in, d_out.data(), 1, cuda::std::plus<>{}, 0, env);
  REQUIRE(c2h::read_single(stream, d_out) == 1);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device reduce not_guaranteed can be tuned", "[reduce][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});

  auto d_in  = block_size_extracting_constant_iterator(1, d_block_size.data());
  auto d_out = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);

  auto env = cuda::std::execution::env{cuda::execution::require(cuda::execution::determinism::not_guaranteed),
                                       cuda::execution::tune(reduce_tuning<target_block_size>{}, unrelated_tuning{})};

  stream.sync();
  device_reduce(d_in, d_out.data(), 1, cuda::std::plus<>{}, 0, env);
  REQUIRE(c2h::read_single(stream, d_out) == 1);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}
CUB_TEST("Device reduce run_to_run can be tuned", "[reduce][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});

  auto d_in  = block_size_extracting_constant_iterator(1, d_block_size.data());
  auto d_out = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);

  auto env = cuda::std::execution::env{cuda::execution::require(cuda::execution::determinism::run_to_run),
                                       cuda::execution::tune(reduce_tuning<target_block_size>{}, unrelated_tuning{})};

  stream.sync();
  device_reduce(d_in, d_out.data(), 1, cuda::std::plus<>{}, 0, env);
  REQUIRE(c2h::read_single(stream, d_out) == 1);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device reduce gpu_to_gpu can be tuned", "[reduce][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});

  auto d_in  = block_size_extracting_constant_iterator(1, d_block_size.data());
  auto d_out = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);

  auto env = cuda::std::execution::env{cuda::execution::require(cuda::execution::determinism::gpu_to_gpu),
                                       cuda::execution::tune(unrelated_tuning{}, reduce_tuning<target_block_size>{})};

  stream.sync();
  device_reduce(d_in, d_out.data(), 1, cuda::std::plus<float>{}, 0, env); // make accum_t float to select RFA
  REQUIRE(c2h::read_single(stream, d_out) == 1);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device Sum can be tuned", "[reduce][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});

  auto d_in  = block_size_extracting_constant_iterator(1, d_block_size.data());
  auto d_out = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);

  auto env = cuda::execution::tune(reduce_tuning<target_block_size>{});

  stream.sync();
  device_reduce_sum(d_in, d_out.data(), 1, env);
  REQUIRE(c2h::read_single(stream, d_out) == 1);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device Min can be tuned", "[reduce][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});

  auto d_in  = block_size_extracting_constant_iterator(42, d_block_size.data());
  auto d_out = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);

  auto env = cuda::execution::tune(reduce_tuning<target_block_size>{});

  stream.sync();
  device_reduce_min(d_in, d_out.data(), 1, env);
  REQUIRE(c2h::read_single(stream, d_out) == 42);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device Max can be tuned", "[reduce][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});

  auto d_in  = block_size_extracting_constant_iterator(42, d_block_size.data());
  auto d_out = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);

  auto env = cuda::execution::tune(reduce_tuning<target_block_size>{});

  stream.sync();
  device_reduce_max(d_in, d_out.data(), 1, env);
  REQUIRE(c2h::read_single(stream, d_out) == 42);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device TransformReduce can be tuned", "[reduce][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});

  auto d_in  = block_size_extracting_constant_iterator(1, d_block_size.data());
  auto d_out = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);

  auto env = cuda::execution::tune(reduce_tuning<target_block_size>{}, unrelated_tuning{});

  stream.sync();
  device_transform_reduce(d_in, d_out.data(), 1, cuda::std::plus<>{}, cuda::std::negate<int>{}, 0, env);
  REQUIRE(c2h::read_single(stream, d_out) == -1);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device ArgMin can be tuned", "[reduce][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  using compare_t                          = block_size_extracting_op<cuda::std::less<>>;
  const compare_t compare_op{d_block_size.data()};

  auto input        = c2h::make_device_buffer<int>(stream, device, {3, 1, 4, 0, 2});
  auto min_output   = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);
  auto index_output = c2h::make_device_buffer<cuda::std::int64_t>(stream, device, 1, cuda::no_init);

  auto env = cuda::execution::tune(reduce_tuning<target_block_size>{});

  stream.sync();
  device_arg_min(input.data(), min_output.data(), index_output.data(), static_cast<int>(input.size()), compare_op, env);
  REQUIRE(c2h::read_single(stream, min_output) == 0);
  REQUIRE(c2h::read_single(stream, index_output) == 3);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device ArgMax can be tuned", "[reduce][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  using compare_t                          = block_size_extracting_op<cuda::std::less<>>;
  const compare_t compare_op{d_block_size.data()};

  auto input        = c2h::make_device_buffer<int>(stream, device, {3, 1, 4, 0, 2});
  auto max_output   = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);
  auto index_output = c2h::make_device_buffer<cuda::std::int64_t>(stream, device, 1, cuda::no_init);

  auto env = cuda::execution::tune(reduce_tuning<target_block_size>{});

  stream.sync();
  device_arg_max(input.data(), max_output.data(), index_output.data(), static_cast<int>(input.size()), compare_op, env);
  REQUIRE(c2h::read_single(stream, max_output) == 4);
  REQUIRE(c2h::read_single(stream, index_output) == 2);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST_CASE("Device reduce not_guaranteed accepts a nondeterministic tuning when falling back to run_to_run",
              "[reduce][device]",
              CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto env = stdexec::env{cuda::execution::require(cuda::execution::determinism::not_guaranteed),
                          cuda::execution::tune(nondeterministic_reduce_tuning{})};

  SECTION("non-plus operator on floats")
  {
    auto input  = c2h::make_device_buffer<float>(stream, device, {3.0f, 1.0f, 4.0f, 0.0f, 2.0f});
    auto output = c2h::make_device_buffer<float>(stream, device, 1, cuda::no_init);
    stream.sync();
    device_reduce(
      input.data(),
      output.data(),
      static_cast<int>(input.size()),
      cuda::maximum<>{},
      cuda::std::numeric_limits<float>::lowest(),
      env);
    REQUIRE(c2h::read_single(stream, output) == 4.0f);
  }
  SECTION("custom operator on ints")
  {
    auto input  = c2h::make_device_buffer<int>(stream, device, {3, 1, 4, 0, 2});
    auto output = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);
    stream.sync();
    device_reduce(input.data(), output.data(), static_cast<int>(input.size()), custom_max_op{}, 0, env);
    REQUIRE(c2h::read_single(stream, output) == 4);
  }
  SECTION("output type differs from the accumulator type")
  {
    auto input  = c2h::make_device_buffer<float>(stream, device, {3.0f, 1.0f, 4.0f, 0.0f, 2.0f});
    auto output = c2h::make_device_buffer<double>(stream, device, 1, cuda::no_init);
    stream.sync();
    device_reduce(input.data(), output.data(), static_cast<int>(input.size()), cuda::std::plus<>{}, 0.0f, env);
    REQUIRE(c2h::read_single(stream, output) == 10.0);
  }
}

CUB_TEST_CASE("Device Min and Max not_guaranteed accept a nondeterministic tuning", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto env = stdexec::env{cuda::execution::require(cuda::execution::determinism::not_guaranteed),
                          cuda::execution::tune(nondeterministic_reduce_tuning{})};

  auto input  = c2h::make_device_buffer<float>(stream, device, {3.0f, 1.0f, 4.0f, 0.0f, 2.0f});
  auto output = c2h::make_device_buffer<float>(stream, device, 1, cuda::no_init);

  SECTION("Min")
  {
    stream.sync();
    device_reduce_min(input.data(), output.data(), static_cast<int>(input.size()), env);
    REQUIRE(c2h::read_single(stream, output) == 0.0f);
  }
  SECTION("Max")
  {
    stream.sync();
    device_reduce_max(input.data(), output.data(), static_cast<int>(input.size()), env);
    REQUIRE(c2h::read_single(stream, output) == 4.0f);
  }
}

CUB_TEST_CASE("Device ArgMin, ArgMax and ArgMinMax not_guaranteed accept a nondeterministic tuning",
              "[reduce][device]",
              CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto env = stdexec::env{cuda::execution::require(cuda::execution::determinism::not_guaranteed),
                          cuda::execution::tune(nondeterministic_reduce_tuning{})};

  auto input        = c2h::make_device_buffer<float>(stream, device, {3.0f, 1.0f, 4.0f, 0.0f, 2.0f});
  const auto n      = static_cast<int>(input.size());
  auto extremum_out = c2h::make_device_buffer<float>(stream, device, 1, cuda::no_init);
  auto index_out    = c2h::make_device_buffer<cuda::std::int64_t>(stream, device, 1, cuda::no_init);

  SECTION("ArgMin")
  {
    stream.sync();
    device_arg_min(input.data(), extremum_out.data(), index_out.data(), n, env);
    REQUIRE(c2h::read_single(stream, extremum_out) == 0.0f);
    REQUIRE(c2h::read_single(stream, index_out) == 3);
  }
  SECTION("ArgMax")
  {
    stream.sync();
    device_arg_max(input.data(), extremum_out.data(), index_out.data(), n, env);
    REQUIRE(c2h::read_single(stream, extremum_out) == 4.0f);
    REQUIRE(c2h::read_single(stream, index_out) == 2);
  }
  SECTION("ArgMinMax")
  {
    auto max_extremum_out = c2h::make_device_buffer<float>(stream, device, 1, cuda::no_init);
    auto max_index_out    = c2h::make_device_buffer<cuda::std::int64_t>(stream, device, 1, cuda::no_init);
    stream.sync();
    device_arg_minmax(
      input.data(), extremum_out.data(), index_out.data(), max_extremum_out.data(), max_index_out.data(), n, env);
    REQUIRE(c2h::read_single(stream, extremum_out) == 0.0f);
    REQUIRE(c2h::read_single(stream, index_out) == 3);
    REQUIRE(c2h::read_single(stream, max_extremum_out) == 4.0f);
    REQUIRE(c2h::read_single(stream, max_index_out) == 2);
  }
}

#endif // TEST_LAUNCH != 1

template <int BlockThreads, int ItemsPerThread = 1>
struct reduce_by_key_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::ReduceByKeyPolicy
  {
    return {cub::ReduceByKeyAlgorithm::lookback,
            {BlockThreads, ItemsPerThread, cub::BLOCK_LOAD_DIRECT, cub::LOAD_DEFAULT, cub::BLOCK_SCAN_WARP_SCANS, {}}};
  }
};

template <int BlockThreads>
struct reduce_by_key_tuning_transform
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability, cub::ReduceByKeyPolicy policy) const
    -> cub::ReduceByKeyPolicy
  {
    policy.lookback.threads_per_block = BlockThreads;
    return policy;
  }
};

using reduce_by_key_block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

#if TEST_LAUNCH != 1

using block_size_extracting_minimum_t = block_size_extracting_op<cuda::minimum<int>>;

CUB_TEST("Device ReduceByKey can be tuned", "[reduce][device]", CUB_SMALL, reduce_by_key_block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_keys_in                           = c2h::make_device_buffer<int>(stream, device, {0, 2, 2, 9, 5, 5, 5, 8});
  auto d_values_in                         = c2h::make_device_buffer<int>(stream, device, {0, 7, 1, 6, 2, 5, 3, 4});
  auto d_unique_out                        = c2h::make_device_buffer<int>(stream, device, 8, cuda::no_init);
  auto d_aggregates_out                    = c2h::make_device_buffer<int>(stream, device, 8, cuda::no_init);
  auto d_num_runs_out                      = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});

  const block_size_extracting_minimum_t reduction_op{d_block_size.data()};
  auto env = cuda::execution::tune(reduce_by_key_tuning<target_block_size>{});

  stream.sync();
  device_reduce_by_key(
    d_keys_in.data(),
    d_unique_out.data(),
    d_values_in.data(),
    d_aggregates_out.data(),
    d_num_runs_out.data(),
    reduction_op,
    static_cast<int>(d_keys_in.size()),
    env);

  REQUIRE(c2h::read_single(stream, d_num_runs_out) == 5);
  const auto expected_keys       = c2h::make_host_buffer<int>(stream, device, {0, 2, 9, 5, 8});
  const auto expected_aggregates = c2h::make_host_buffer<int>(stream, device, {0, 1, 6, 2, 4});
  check_output_prefix(stream, device, d_unique_out, expected_keys);
  check_output_prefix(stream, device, d_aggregates_out, expected_aggregates);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST_CASE("Device ReduceByKey tuning can be modified", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = 64; // must be different from any value we use in production
  auto d_keys_in                           = c2h::make_device_buffer<int>(stream, device, {0, 2, 2, 9, 5, 5, 5, 8});
  auto d_values_in                         = c2h::make_device_buffer<int>(stream, device, {0, 7, 1, 6, 2, 5, 3, 4});
  auto d_unique_out                        = c2h::make_device_buffer<int>(stream, device, 8, cuda::no_init);
  auto d_aggregates_out                    = c2h::make_device_buffer<int>(stream, device, 8, cuda::no_init);
  auto d_num_runs_out                      = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});

  const block_size_extracting_minimum_t reduction_op{d_block_size.data()};
  auto env = cuda::execution::tune(reduce_by_key_tuning_transform<target_block_size>{});

  {
    stream.sync();
    device_reduce_by_key(
      d_keys_in.data(),
      d_unique_out.data(),
      d_values_in.data(),
      d_aggregates_out.data(),
      d_num_runs_out.data(),
      reduction_op,
      static_cast<int>(d_keys_in.size()),
      stdexec::env<>{});
    const unsigned int default_block_size = c2h::read_single(stream, d_block_size);
    REQUIRE(default_block_size != target_block_size);
    cuda::fill_bytes(stream, d_block_size, 0);
  }

  stream.sync();
  device_reduce_by_key(
    d_keys_in.data(),
    d_unique_out.data(),
    d_values_in.data(),
    d_aggregates_out.data(),
    d_num_runs_out.data(),
    reduction_op,
    static_cast<int>(d_keys_in.size()),
    env);
  REQUIRE(c2h::read_single(stream, d_num_runs_out) == 5);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

// The agent's temporary storage for 32 threads x 100 items exceeds the static shared memory limit, so the kernel falls
// back to a larger block of 64 threads, which its launch bounds must allow
CUB_TEST_CASE("Device ReduceByKey can be tuned with a small block size that needs the fallback policy",
              "[reduce][device]",
              CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  constexpr int num_items = 10000;
  constexpr int run_size  = 7;
  constexpr int num_runs  = cuda::ceil_div(num_items, run_size);

  // runs of run_size equal keys 0, 0, ..., 1, 1, ..., 2, ...; all values are 1
  auto d_keys_in = c2h::make_device_buffer<long long>(stream, num_items, cuda::no_init);
  thrust::tabulate(c2h::device_policy.on(stream.get()),
                   d_keys_in.data(),
                   d_keys_in.data() + d_keys_in.size(),
                   thrust::placeholders::_1 / run_size);
  auto d_values_in      = c2h::make_device_buffer<double>(stream, num_items, 1.0);
  auto d_unique_out     = c2h::make_device_buffer<long long>(stream, num_runs, cuda::no_init);
  auto d_aggregates_out = c2h::make_device_buffer<double>(stream, num_runs, cuda::no_init);
  auto d_num_runs_out   = c2h::make_device_buffer<int>(stream, 1, cuda::no_init);

  auto env = stdexec::env{cuda::stream_ref{stream}, cuda::execution::tune(reduce_by_key_tuning<32, 100>{})};

  device_reduce_by_key(
    d_keys_in.data(),
    d_unique_out.data(),
    d_values_in.data(),
    d_aggregates_out.data(),
    d_num_runs_out.data(),
    cuda::std::plus<>{},
    num_items,
    env);

  REQUIRE(c2h::read_single(stream, d_num_runs_out) == num_runs);
  auto expected_keys = c2h::make_host_buffer<long long>(stream, num_runs, cuda::no_init);
  cuda::std::iota(expected_keys.begin(), expected_keys.end(), 0LL);
  auto expected_aggregates   = c2h::make_host_buffer<double>(stream, num_runs, run_size);
  expected_aggregates.back() = num_items - ((num_runs - 1) * run_size); // the last run is shorter
  REQUIRE_THAT(d_unique_out, c2h::Equals(stream, expected_keys));
  REQUIRE_THAT(d_aggregates_out, c2h::Equals(stream, expected_aggregates));
}

#endif // TEST_LAUNCH != 1

using requirements =
  c2h::type_list<cuda::execution::determinism::gpu_to_gpu_t,
                 cuda::execution::determinism::run_to_run_t,
                 cuda::execution::determinism::not_guaranteed_t>;

CUB_TEST("Device reduce uses environment", "[reduce][device]", CUB_SMALL, requirements)
{
  const auto stream = c2h::make_current_device_stream();

  using determinism_t = c2h::get<0, TestType>;
  using accumulator_t = float;
  using op_t          = cuda::std::plus<>;
  using num_items_t   = int;
  using offset_t      = cub::detail::choose_offset_t<num_items_t>;
  using transform_t   = cuda::std::identity;
  using init_value_t  = accumulator_t;

  num_items_t num_items = GENERATE(1 << 4, 1 << 24);
  auto d_in             = cuda::constant_iterator(1.0f);
  auto d_out            = c2h::make_device_buffer<accumulator_t>(stream, 1, cuda::no_init);

  init_value_t init = 0;
  size_t expected_bytes_allocated{};

  // MSVC yields a reference type for `decltype(d_in)`/`decltype(d_out.data())` when `d_in`/`d_out` are ODR-used inside
  // the by-reference-capturing lambda, which would select different kernel instantiations than the ones dispatched
  // below. Hoisting the decltypes and decay-ing them to make the aliases position independent.
  using input_it_t  = ::cuda::std::decay_t<decltype(d_in)>;
  using output_it_t = ::cuda::std::decay_t<decltype(d_out.data())>;

  // To check if a given algorithm implementation is used, we check if associated kernels are invoked.
  auto kernels = [&]() {
    if constexpr (cuda::std::is_same_v<determinism_t, cuda::execution::determinism::run_to_run_t>)
    {
      REQUIRE(
        cudaSuccess
        == cub::DeviceReduce::Reduce(nullptr, expected_bytes_allocated, d_in, d_out.data(), num_items, op_t{}, init));

      using policy_t = cub::detail::reduce::policy_selector_from_types<accumulator_t, offset_t, op_t>;
      return cuda::std::array<void*, 3>{
        reinterpret_cast<void*>(
          cub::detail::reduce::DeviceReduceSingleTileKernel<
            policy_t,
            input_it_t,
            output_it_t,
            offset_t,
            op_t,
            init_value_t,
            accumulator_t,
            transform_t>),
        reinterpret_cast<void*>(
          cub::detail::reduce::DeviceReduceKernel<
            policy_t,
            /* StableReductionOrder */ true,
            input_it_t,
            accumulator_t*,
            offset_t,
            offset_t,
            op_t,
            accumulator_t,
            init_value_t,
            transform_t>),
        reinterpret_cast<void*>(
          cub::detail::reduce::DeviceReduceSingleTileKernel<
            policy_t,
            accumulator_t*,
            output_it_t,
            int, // always used with int offset
            op_t,
            init_value_t,
            accumulator_t>)};
    }
    else if constexpr (cub::detail::is_non_deterministic_v<determinism_t>)
    {
      using policy_t =
        cub::detail::reduce::policy_selector_from_types<accumulator_t, offset_t, op_t, __determinism_t::__not_guaranteed>;
      auto* raw_ptr = d_out.data();

      REQUIRE(
        cudaSuccess
        == cub::detail::reduce::dispatch<cub::detail::use_default, /* StableReductionOrder */ false>(
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
        cub::detail::reduce::DeviceReduceKernel<
          policy_t,
          /* StableReductionOrder */ false,
          input_it_t,
          decltype(raw_ptr),
          offset_t,
          offset_t,
          op_t,
          accumulator_t,
          init_value_t,
          transform_t>)};
    }
    else
    {
      using deterministic_add_t = cub::detail::rfa::deterministic_sum_t<accumulator_t>;
      using reduction_op_t      = deterministic_add_t;
      using policy_t            = cub::detail::reduce::
        policy_selector_from_types<accumulator_t, offset_t, reduction_op_t, __determinism_t::__gpu_to_gpu>;
      using deterministic_accum_t = deterministic_add_t::DeterministicAcc;

      REQUIRE(
        cudaSuccess
        == cub::detail::rfa::dispatch<input_it_t, output_it_t, offset_t, init_value_t, transform_t, accumulator_t>(
          nullptr, expected_bytes_allocated, d_in, d_out.data(), num_items, init));

      auto k1 = cub::detail::reduce::DeterministicDeviceReduceSingleTileKernel<
        policy_t,
        input_it_t,
        output_it_t,
        reduction_op_t,
        init_value_t,
        deterministic_accum_t,
        transform_t>;
      auto k2 = cub::detail::reduce::
        DeterministicDeviceReduceKernel<policy_t, input_it_t, int, reduction_op_t, deterministic_accum_t, transform_t>;
      auto k3 = cub::detail::reduce::DeterministicDeviceReduceSingleTileKernel<
        policy_t,
        deterministic_accum_t*,
        output_it_t,
        reduction_op_t,
        init_value_t,
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
  auto env = stdexec::env{
    cuda::stream_ref{stream},
    cuda::execution::require(determinism_t{}), // determinism
    allowed_kernels(kernels), // allowed kernels for the given determinism
    expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_reduce(d_in, d_out.data(), num_items, op_t{}, init, env);

  REQUIRE(c2h::read_single(stream, d_out) == num_items);

#if TEST_LAUNCH == 0
  // The two-phase overload uses the same kernels for the requested determinism
  const kernel_scope two_phase_kernels{kernels};
  const auto requirements   = stdexec::env{cuda::stream_ref{stream}, cuda::execution::require(determinism_t{})};
  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess
          == cub::DeviceReduce::Reduce(
            nullptr, temp_storage_bytes, d_in, d_out.data(), num_items, op_t{}, init, requirements));
  auto temp_storage = c2h::make_device_buffer<cuda::std::uint8_t>(stream, temp_storage_bytes, cuda::no_init);
  cuda::fill_bytes(stream, d_out, 0x5a);
  REQUIRE(cudaSuccess
          == cub::DeviceReduce::Reduce(
            temp_storage.data(), temp_storage_bytes, d_in, d_out.data(), num_items, op_t{}, init, requirements));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(c2h::read_single(stream, d_out) == num_items);
#endif // TEST_LAUNCH == 0
}

CUB_TEST("Device sum uses environment", "[reduce][device]", CUB_SMALL, requirements)
{
  const auto stream = c2h::make_current_device_stream();

  using determinism_t = c2h::get<0, TestType>;
  using accumulator_t = float;
  using op_t          = cuda::std::plus<>;
  using num_items_t   = int;
  using offset_t      = cub::detail::choose_offset_t<num_items_t>;
  using transform_t   = cuda::std::identity;
  using init_value_t  = accumulator_t;

  num_items_t num_items = GENERATE(1 << 4, 1 << 24);
  auto d_in             = cuda::constant_iterator(1.0f);
  auto d_out            = c2h::make_device_buffer<accumulator_t>(stream, 1, cuda::no_init);

  // MSVC yields a reference type for `decltype(d_in)`/`decltype(d_out.data())` when `d_in`/`d_out` are ODR-used inside
  // the by-reference-capturing lambda, which would select different kernel instantiations than the ones dispatched
  // below. Hoisting the decltypes and decay-ing them to make the aliases position independent.
  using input_it_t  = ::cuda::std::decay_t<decltype(d_in)>;
  using output_it_t = ::cuda::std::decay_t<decltype(d_out.data())>;

  [[maybe_unused]] const init_value_t init = 0;
  size_t expected_bytes_allocated{};

  // To check if a given algorithm implementation is used, we check if associated kernels are invoked.
  auto kernels = [&]() {
    if constexpr (cuda::std::is_same_v<determinism_t, cuda::execution::determinism::run_to_run_t>)
    {
      REQUIRE(cudaSuccess == cub::DeviceReduce::Sum(nullptr, expected_bytes_allocated, d_in, d_out.data(), num_items));

      using policy_t = cub::detail::reduce::policy_selector_from_types<accumulator_t, offset_t, op_t>;
      return cuda::std::array<void*, 3>{
        reinterpret_cast<void*>(
          cub::detail::reduce::DeviceReduceSingleTileKernel<
            policy_t,
            input_it_t,
            output_it_t,
            offset_t,
            op_t,
            init_value_t,
            accumulator_t,
            transform_t>),
        reinterpret_cast<void*>(
          cub::detail::reduce::DeviceReduceKernel<
            policy_t,
            /* StableReductionOrder */ true,
            input_it_t,
            accumulator_t*,
            offset_t,
            offset_t,
            op_t,
            accumulator_t,
            init_value_t,
            transform_t>),
        reinterpret_cast<void*>(
          cub::detail::reduce::DeviceReduceSingleTileKernel<
            policy_t,
            accumulator_t*,
            output_it_t,
            int, // always used with int offset
            op_t,
            init_value_t,
            accumulator_t>)};
    }
    else if constexpr (cub::detail::is_non_deterministic_v<determinism_t>)
    {
      using policy_t =
        cub::detail::reduce::policy_selector_from_types<accumulator_t, offset_t, op_t, __determinism_t::__not_guaranteed>;
      auto* raw_ptr = d_out.data();

      REQUIRE(
        cudaSuccess
        == cub::detail::reduce::dispatch<cub::detail::use_default, /* StableReductionOrder */ false>(
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
        cub::detail::reduce::DeviceReduceKernel<
          policy_t,
          /* StableReductionOrder */ false,
          input_it_t,
          decltype(raw_ptr),
          offset_t,
          offset_t,
          op_t,
          accumulator_t,
          init_value_t,
          transform_t>)};
    }
    else
    {
      using deterministic_add_t = cub::detail::rfa::deterministic_sum_t<accumulator_t>;
      using reduction_op_t      = deterministic_add_t;
      using policy_t            = cub::detail::reduce::
        policy_selector_from_types<accumulator_t, offset_t, reduction_op_t, __determinism_t::__gpu_to_gpu>;
      using deterministic_accum_t = deterministic_add_t::DeterministicAcc;

      REQUIRE(
        cudaSuccess
        == cub::detail::rfa::dispatch<input_it_t, output_it_t, offset_t, init_value_t, transform_t, accumulator_t>(
          nullptr, expected_bytes_allocated, d_in, d_out.data(), num_items, init));

      auto k1 = cub::detail::reduce::DeterministicDeviceReduceSingleTileKernel<
        policy_t,
        input_it_t,
        output_it_t,
        reduction_op_t,
        init_value_t,
        deterministic_accum_t,
        transform_t>;
      auto k2 = cub::detail::reduce::
        DeterministicDeviceReduceKernel<policy_t, input_it_t, int, reduction_op_t, deterministic_accum_t, transform_t>;
      auto k3 = cub::detail::reduce::DeterministicDeviceReduceSingleTileKernel<
        policy_t,
        deterministic_accum_t*,
        output_it_t,
        reduction_op_t,
        init_value_t,
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
  auto env = stdexec::env{
    cuda::stream_ref{stream},
    cuda::execution::require(determinism_t{}), // determinism
    allowed_kernels(kernels), // allowed kernels for the given determinism
    expected_allocation_size(expected_bytes_allocated)}; // temp storage size

  device_reduce_sum(d_in, d_out.data(), num_items, env);

  REQUIRE(c2h::read_single(stream, d_out) == num_items);

#if TEST_LAUNCH == 0
  // The two-phase overload uses the same kernels for the requested determinism
  const kernel_scope two_phase_kernels{kernels};
  const auto requirements   = stdexec::env{cuda::stream_ref{stream}, cuda::execution::require(determinism_t{})};
  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess == cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, d_in, d_out.data(), num_items, requirements));
  auto temp_storage = c2h::make_device_buffer<cuda::std::uint8_t>(stream, temp_storage_bytes, cuda::no_init);
  cuda::fill_bytes(stream, d_out, 0x5a);
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::Sum(temp_storage.data(), temp_storage_bytes, d_in, d_out.data(), num_items, requirements));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(c2h::read_single(stream, d_out) == num_items);
#endif // TEST_LAUNCH == 0
}

CUB_TEST("Device reduce not_guaranteed falls back when output type differs from accumulator",
         "[reduce][device]",
         CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  using input_t       = cuda::std::uint8_t;
  using output_t      = cuda::std::uint8_t;
  using accumulator_t = int;
  using op_t          = cuda::std::plus<>;
  using init_value_t  = input_t;
  using num_items_t   = int;
  using offset_t      = cub::detail::choose_offset_t<num_items_t>;
  using transform_t   = cuda::std::identity;

  const auto d_in             = c2h::make_device_buffer<input_t>(stream, {0, 1, 2, 3});
  auto d_out                  = c2h::make_device_buffer<output_t>(stream, 1, cuda::no_init);
  const num_items_t num_items = static_cast<num_items_t>(d_in.size());
  const init_value_t init{};
  size_t expected_bytes_allocated{};

  REQUIRE(cudaSuccess
          == cub::DeviceReduce::Reduce(
            nullptr, expected_bytes_allocated, d_in.data(), d_out.data(), num_items, op_t{}, init));

  using policy_t = cub::detail::reduce::policy_selector_from_types<accumulator_t, offset_t, op_t>;
  auto kernels   = cuda::std::array<void*, 3>{
    reinterpret_cast<void*>(
      cub::detail::reduce::DeviceReduceSingleTileKernel<
        policy_t,
        decltype(d_in.data()),
        decltype(d_out.data()),
        offset_t,
        op_t,
        init_value_t,
        accumulator_t,
        transform_t>),
    reinterpret_cast<void*>(
      cub::detail::reduce::DeviceReduceKernel<
        policy_t,
        /* StableReductionOrder */ true,
        decltype(d_in.data()),
        accumulator_t*,
        offset_t,
        offset_t,
        op_t,
        accumulator_t,
        init_value_t,
        transform_t>),
    reinterpret_cast<void*>(
      cub::detail::reduce::DeviceReduceSingleTileKernel<
        policy_t,
        accumulator_t*,
        decltype(d_out.data()),
        int, // always used with int offset
        op_t,
        init_value_t,
        accumulator_t>)};

  auto env = stdexec::env{
    cuda::stream_ref{stream},
    cuda::execution::require(cuda::execution::determinism::not_guaranteed),
    allowed_kernels(kernels),
    expected_allocation_size(expected_bytes_allocated)};

  device_reduce(d_in.data(), d_out.data(), num_items, op_t{}, init, env);

  REQUIRE(c2h::read_single(stream, d_out) == output_t{6});
}

CUB_TEST("Device sum not_guaranteed falls back when output type differs from accumulator", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  using input_t       = cuda::std::uint8_t;
  using output_t      = cuda::std::uint8_t;
  using accumulator_t = int;
  using op_t          = cuda::std::plus<>;
  using init_value_t  = output_t;
  using num_items_t   = int;
  using offset_t      = cub::detail::choose_offset_t<num_items_t>;
  using transform_t   = cuda::std::identity;

  auto d_in                   = c2h::make_device_buffer<input_t>(stream, {0, 1, 2, 3});
  auto d_out                  = c2h::make_device_buffer<output_t>(stream, 1, cuda::no_init);
  const num_items_t num_items = static_cast<num_items_t>(d_in.size());
  size_t expected_bytes_allocated{};

  REQUIRE(
    cudaSuccess == cub::DeviceReduce::Sum(nullptr, expected_bytes_allocated, d_in.data(), d_out.data(), num_items));

  using policy_t = cub::detail::reduce::policy_selector_from_types<accumulator_t, offset_t, op_t>;
  auto kernels   = cuda::std::array<void*, 3>{
    reinterpret_cast<void*>(
      cub::detail::reduce::DeviceReduceSingleTileKernel<
        policy_t,
        decltype(d_in.data()),
        decltype(d_out.data()),
        offset_t,
        op_t,
        init_value_t,
        accumulator_t,
        transform_t>),
    reinterpret_cast<void*>(
      cub::detail::reduce::DeviceReduceKernel<
        policy_t,
        /* StableReductionOrder */ true,
        decltype(d_in.data()),
        accumulator_t*,
        offset_t,
        offset_t,
        op_t,
        accumulator_t,
        init_value_t,
        transform_t>),
    reinterpret_cast<void*>(
      cub::detail::reduce::DeviceReduceSingleTileKernel<
        policy_t,
        accumulator_t*,
        decltype(d_out.data()),
        int, // always used with int offset
        op_t,
        init_value_t,
        accumulator_t>)};

  auto env = stdexec::env{
    cuda::stream_ref{stream},
    cuda::execution::require(cuda::execution::determinism::not_guaranteed),
    allowed_kernels(kernels),
    expected_allocation_size(expected_bytes_allocated)};

  device_reduce_sum(d_in.data(), d_out.data(), num_items, env);

  REQUIRE(c2h::read_single(stream, d_out) == output_t{6});
}

#if TEST_LAUNCH == 0

CUB_TEST_CASE("Device Min works with default environment", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto input  = c2h::make_device_buffer<float>(stream, device, {3.0f, 1.0f, 4.0f, 0.0f, 2.0f});
  auto output = c2h::make_device_buffer<float>(stream, device, 1, cuda::no_init);

  auto error = cub::DeviceReduce::Min(input.data(), output.data(), static_cast<int>(input.size()));
  REQUIRE(error == cudaSuccess);
  REQUIRE(c2h::read_single(stream, output) == 0.0f);
}

CUB_TEST_CASE("Device Max works with default environment", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto input  = c2h::make_device_buffer<float>(stream, device, {3.0f, 1.0f, 4.0f, 0.0f, 2.0f});
  auto output = c2h::make_device_buffer<float>(stream, device, 1, cuda::no_init);

  auto error = cub::DeviceReduce::Max(input.data(), output.data(), static_cast<int>(input.size()));
  REQUIRE(error == cudaSuccess);
  REQUIRE(c2h::read_single(stream, output) == 4.0f);
}

CUB_TEST_CASE("Device TransformReduce works with default environment", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto d_in  = c2h::make_device_buffer<int>(stream, device, {1, 2, 3, 4});
  auto d_out = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);

  auto negate = cuda::std::negate<int>{};

  REQUIRE(cudaSuccess
          == cub::DeviceReduce::TransformReduce(
            d_in.data(), d_out.data(), static_cast<int>(d_in.size()), cuda::std::plus<int>{}, negate, 0));

  REQUIRE(c2h::read_single(stream, d_out) == -10);
}

CUB_TEST_CASE("Device ReduceByKey works with default environment", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto d_keys_in        = c2h::make_device_buffer<int>(stream, device, {0, 2, 2, 9, 5, 5, 5, 8});
  auto d_values_in      = c2h::make_device_buffer<int>(stream, device, {0, 7, 1, 6, 2, 5, 3, 4});
  auto d_unique_out     = c2h::make_device_buffer<int>(stream, device, 8, cuda::no_init);
  auto d_aggregates_out = c2h::make_device_buffer<int>(stream, device, 8, cuda::no_init);
  auto d_num_runs_out   = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);

  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::ReduceByKey(
      d_keys_in.data(),
      d_unique_out.data(),
      d_values_in.data(),
      d_aggregates_out.data(),
      d_num_runs_out.data(),
      cuda::minimum<int>{},
      static_cast<int>(d_keys_in.size())));

  REQUIRE(c2h::read_single(stream, d_num_runs_out) == 5);
  const auto expected_keys       = c2h::make_host_buffer<int>(stream, device, {0, 2, 9, 5, 8});
  const auto expected_aggregates = c2h::make_host_buffer<int>(stream, device, {0, 1, 6, 2, 4});
  check_output_prefix(stream, device, d_unique_out, expected_keys);
  check_output_prefix(stream, device, d_aggregates_out, expected_aggregates);
}

CUB_TEST_CASE("Device ArgMin works with default environment", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto input        = c2h::make_device_buffer<float>(stream, device, {3.0f, 1.0f, 4.0f, 0.0f, 2.0f});
  auto min_output   = c2h::make_device_buffer<float>(stream, device, 1, cuda::no_init);
  auto index_output = c2h::make_device_buffer<cuda::std::int64_t>(stream, device, 1, cuda::no_init);

  auto error =
    cub::DeviceReduce::ArgMin(input.data(), min_output.data(), index_output.data(), static_cast<int>(input.size()));
  REQUIRE(error == cudaSuccess);
  REQUIRE(c2h::read_single(stream, min_output) == 0.0f);
  REQUIRE(c2h::read_single(stream, index_output) == 3);
}

CUB_TEST_CASE("Device ArgMax works with default environment", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto input        = c2h::make_device_buffer<float>(stream, device, {3.0f, 1.0f, 4.0f, 0.0f, 2.0f});
  auto max_output   = c2h::make_device_buffer<float>(stream, device, 1, cuda::no_init);
  auto index_output = c2h::make_device_buffer<cuda::std::int64_t>(stream, device, 1, cuda::no_init);

  auto error =
    cub::DeviceReduce::ArgMax(input.data(), max_output.data(), index_output.data(), static_cast<int>(input.size()));
  REQUIRE(error == cudaSuccess);
  REQUIRE(c2h::read_single(stream, max_output) == 4.0f);
  REQUIRE(c2h::read_single(stream, index_output) == 2);
}

CUB_TEST_CASE("Device ArgMin with compare_op works with default environment", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto input        = c2h::make_device_buffer<float>(stream, device, {3.0f, 1.0f, 4.0f, 0.0f, 2.0f});
  auto min_output   = c2h::make_device_buffer<float>(stream, device, 1, cuda::no_init);
  auto index_output = c2h::make_device_buffer<cuda::std::int64_t>(stream, device, 1, cuda::no_init);

  auto error = cub::DeviceReduce::ArgMin(
    input.data(), min_output.data(), index_output.data(), static_cast<int>(input.size()), cuda::std::less{});
  REQUIRE(error == cudaSuccess);
  REQUIRE(c2h::read_single(stream, min_output) == 0.0f);
  REQUIRE(c2h::read_single(stream, index_output) == 3);
}

CUB_TEST_CASE("Device ArgMax with compare_op works with default environment", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  auto input        = c2h::make_device_buffer<float>(stream, device, {3.0f, 1.0f, 4.0f, 0.0f, 2.0f});
  auto max_output   = c2h::make_device_buffer<float>(stream, device, 1, cuda::no_init);
  auto index_output = c2h::make_device_buffer<cuda::std::int64_t>(stream, device, 1, cuda::no_init);

  auto error = cub::DeviceReduce::ArgMax(
    input.data(), max_output.data(), index_output.data(), static_cast<int>(input.size()), cuda::std::less{});
  REQUIRE(error == cudaSuccess);
  REQUIRE(c2h::read_single(stream, max_output) == 4.0f);
  REQUIRE(c2h::read_single(stream, index_output) == 2);
}

#endif

CUB_TEST("Device TransformReduce uses environment", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  auto d_in  = c2h::make_device_buffer<int>(stream, {1, 2, 3, 4});
  auto d_out = c2h::make_device_buffer<int>(stream, 1, cuda::no_init);

  auto negate = cuda::std::negate<int>{};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::TransformReduce(
      nullptr,
      expected_bytes_allocated,
      d_in.data(),
      d_out.data(),
      static_cast<int>(d_in.size()),
      cuda::std::plus<int>{},
      negate,
      0));

  auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_transform_reduce(
    d_in.data(), d_out.data(), static_cast<int>(d_in.size()), cuda::std::plus<int>{}, negate, 0, env);

  REQUIRE(c2h::read_single(stream, d_out) == -10);
}

CUB_TEST("Device Min uses environment", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  auto input  = c2h::make_device_buffer<float>(stream, {3.0f, 1.0f, 4.0f, 0.0f, 2.0f});
  auto output = c2h::make_device_buffer<float>(stream, 1, cuda::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceReduce::Min(
            nullptr, expected_bytes_allocated, input.data(), output.data(), static_cast<int>(input.size())));

  auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_reduce_min(input.data(), output.data(), static_cast<int>(input.size()), env);

  REQUIRE(c2h::read_single(stream, output) == 0.0f);
}

CUB_TEST("Device Max uses environment", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  auto input  = c2h::make_device_buffer<float>(stream, {3.0f, 1.0f, 4.0f, 0.0f, 2.0f});
  auto output = c2h::make_device_buffer<float>(stream, 1, cuda::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(cudaSuccess
          == cub::DeviceReduce::Max(
            nullptr, expected_bytes_allocated, input.data(), output.data(), static_cast<int>(input.size())));

  auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_reduce_max(input.data(), output.data(), static_cast<int>(input.size()), env);

  REQUIRE(c2h::read_single(stream, output) == 4.0f);
}

CUB_TEST("Device ReduceByKey uses environment", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();
  const auto device = c2h::current_device();

  auto d_keys_in        = c2h::make_device_buffer<int>(stream, {0, 2, 2, 9, 5, 5, 5, 8});
  auto d_values_in      = c2h::make_device_buffer<int>(stream, {0, 7, 1, 6, 2, 5, 3, 4});
  auto d_unique_out     = c2h::make_device_buffer<int>(stream, 8, cuda::no_init);
  auto d_aggregates_out = c2h::make_device_buffer<int>(stream, 8, cuda::no_init);
  auto d_num_runs_out   = c2h::make_device_buffer<int>(stream, 1, cuda::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::ReduceByKey(
      nullptr,
      expected_bytes_allocated,
      d_keys_in.data(),
      d_unique_out.data(),
      d_values_in.data(),
      d_aggregates_out.data(),
      d_num_runs_out.data(),
      cuda::minimum<int>{},
      static_cast<int>(d_keys_in.size())));

  auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_reduce_by_key(
    d_keys_in.data(),
    d_unique_out.data(),
    d_values_in.data(),
    d_aggregates_out.data(),
    d_num_runs_out.data(),
    cuda::minimum<int>{},
    static_cast<int>(d_keys_in.size()),
    env);

  REQUIRE(c2h::read_single(stream, d_num_runs_out) == 5);
  const auto expected_keys       = c2h::make_host_buffer<int>(stream, {0, 2, 9, 5, 8});
  const auto expected_aggregates = c2h::make_host_buffer<int>(stream, {0, 1, 6, 2, 4});
  check_output_prefix(stream, device, d_unique_out, expected_keys);
  check_output_prefix(stream, device, d_aggregates_out, expected_aggregates);
}

CUB_TEST("Device ArgMin uses environment", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  auto input        = c2h::make_device_buffer<float>(stream, {3.0f, 1.0f, 4.0f, 0.0f, 2.0f});
  auto min_output   = c2h::make_device_buffer<float>(stream, 1, cuda::no_init);
  auto index_output = c2h::make_device_buffer<cuda::std::int64_t>(stream, 1, cuda::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::ArgMin(
      nullptr,
      expected_bytes_allocated,
      input.data(),
      min_output.data(),
      index_output.data(),
      static_cast<int>(input.size())));

  auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_arg_min(input.data(), min_output.data(), index_output.data(), static_cast<int>(input.size()), env);

  REQUIRE(c2h::read_single(stream, min_output) == 0.0f);
  REQUIRE(c2h::read_single(stream, index_output) == 3);
}

CUB_TEST("Device ArgMin with compare_op uses environment", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  auto input        = c2h::make_device_buffer<float>(stream, {3.0f, 1.0f, 4.0f, 0.0f, 2.0f});
  auto min_output   = c2h::make_device_buffer<float>(stream, 1, cuda::no_init);
  auto index_output = c2h::make_device_buffer<cuda::std::int64_t>(stream, 1, cuda::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::ArgMin(
      nullptr,
      expected_bytes_allocated,
      input.data(),
      min_output.data(),
      index_output.data(),
      static_cast<int>(input.size()),
      cuda::std::less{}));

  auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_arg_min(
    input.data(), min_output.data(), index_output.data(), static_cast<int>(input.size()), cuda::std::less{}, env);

  REQUIRE(c2h::read_single(stream, min_output) == 0.0f);
  REQUIRE(c2h::read_single(stream, index_output) == 3);
}

CUB_TEST("Device ArgMax uses environment", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  auto input        = c2h::make_device_buffer<float>(stream, {3.0f, 1.0f, 4.0f, 0.0f, 2.0f});
  auto max_output   = c2h::make_device_buffer<float>(stream, 1, cuda::no_init);
  auto index_output = c2h::make_device_buffer<cuda::std::int64_t>(stream, 1, cuda::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::ArgMax(
      nullptr,
      expected_bytes_allocated,
      input.data(),
      max_output.data(),
      index_output.data(),
      static_cast<int>(input.size())));

  auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_arg_max(input.data(), max_output.data(), index_output.data(), static_cast<int>(input.size()), env);

  REQUIRE(c2h::read_single(stream, max_output) == 4.0f);
  REQUIRE(c2h::read_single(stream, index_output) == 2);
}

CUB_TEST("Device ArgMax with compare_op uses environment", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  auto input        = c2h::make_device_buffer<float>(stream, {3.0f, 1.0f, 4.0f, 0.0f, 2.0f});
  auto max_output   = c2h::make_device_buffer<float>(stream, 1, cuda::no_init);
  auto index_output = c2h::make_device_buffer<cuda::std::int64_t>(stream, 1, cuda::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::ArgMax(
      nullptr,
      expected_bytes_allocated,
      input.data(),
      max_output.data(),
      index_output.data(),
      static_cast<int>(input.size()),
      cuda::std::less{}));

  auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_arg_max(
    input.data(), max_output.data(), index_output.data(), static_cast<int>(input.size()), cuda::std::less{}, env);

  REQUIRE(c2h::read_single(stream, max_output) == 4.0f);
  REQUIRE(c2h::read_single(stream, index_output) == 2);
}

CUB_TEST("cub::DeviceReduce::Reduce allows no_init in env overloads", "[reduce][env]", CUB_SMALL)
{
  const auto stream = c2h::make_current_device_stream();

  auto input  = c2h::make_device_buffer<int>(stream, {1, 2, 3, 4, 5});
  auto output = c2h::make_device_buffer<int>(stream, 1, cuda::no_init);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceReduce::Reduce(
      nullptr,
      expected_bytes_allocated,
      input.data(),
      output.data(),
      input.size(),
      cuda::std::plus<>{},
      cub::detail::reduce::no_init));

  auto env = stdexec::env{cuda::stream_ref{stream}, expected_allocation_size(expected_bytes_allocated)};

  device_reduce(input.data(), output.data(), input.size(), cuda::std::plus<>{}, cub::detail::reduce::no_init, env);
  REQUIRE(c2h::read_single(stream, output) == 15);
}

#if TEST_LAUNCH == 0

template <class EnvT, class TwoPhaseFn>
void run_two_phase(const EnvT& env, TwoPhaseFn two_phase)
{
  const auto stream = cuda::__call_or(cuda::get_stream, c2h::default_stream_ref(), env);
  const stream_scope scope{stream.get()};

  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == two_phase(nullptr, temp_storage_bytes));

  auto temp_storage =
    c2h::make_device_buffer<cuda::std::uint8_t>(stream, c2h::current_device(), temp_storage_bytes, cuda::no_init);
  REQUIRE(cudaSuccess == two_phase(temp_storage.data(), temp_storage_bytes));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

// The launch wrappers copy their arguments, which the non-copyable streams of test_with_custom_streams do not allow
CUB_TEST_CASE("Device Reduce two-phase works with custom streams", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_in = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9});
  auto d_out      = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);

  stream.sync();
  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes, d_in.data(), d_out.data(), 7, cuda::std::plus<>{}, 10, env);
    });
    REQUIRE(c2h::read_single(stream, d_out) == 48);
  });
}

CUB_TEST_CASE("Device Sum two-phase works with custom streams", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_in = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9});
  auto d_out      = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);

  stream.sync();
  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in.data(), d_out.data(), 7, env);
    });
    REQUIRE(c2h::read_single(stream, d_out) == 38);
  });
}

CUB_TEST_CASE("Device Min two-phase works with custom streams", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_in = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9});
  auto d_out      = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);

  stream.sync();
  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in.data(), d_out.data(), 7, env);
    });
    REQUIRE(c2h::read_single(stream, d_out) == 0);
  });
}

CUB_TEST_CASE("Device Max two-phase works with custom streams", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_in = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9});
  auto d_out      = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);

  stream.sync();
  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in.data(), d_out.data(), 7, env);
    });
    REQUIRE(c2h::read_single(stream, d_out) == 9);
  });
}

CUB_TEST_CASE("Device TransformReduce two-phase works with custom streams", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_in = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9});
  auto d_out      = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);

  stream.sync();
  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceReduce::TransformReduce(
        d_temp_storage,
        temp_storage_bytes,
        d_in.data(),
        d_out.data(),
        7,
        cuda::std::plus<>{},
        cuda::std::negate<int>{},
        0,
        env);
    });
    REQUIRE(c2h::read_single(stream, d_out) == -38);
  });
}

CUB_TEST_CASE("Device ReduceByKey two-phase works with custom streams", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_keys_in   = c2h::make_device_buffer<int>(stream, device, {0, 2, 2, 9, 5, 5, 5, 8});
  const auto d_values_in = c2h::make_device_buffer<int>(stream, device, {0, 7, 1, 6, 2, 5, 3, 4});
  auto d_unique_out      = c2h::make_device_buffer<int>(stream, device, 5, cuda::no_init);
  auto d_aggregates_out  = c2h::make_device_buffer<int>(stream, device, 5, cuda::no_init);
  auto d_num_runs_out    = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);

  stream.sync();
  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceReduce::ReduceByKey(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in.data(),
        d_unique_out.data(),
        d_values_in.data(),
        d_aggregates_out.data(),
        d_num_runs_out.data(),
        cuda::minimum<>{},
        8,
        env);
    });
    REQUIRE(c2h::read_single(stream, d_num_runs_out) == 5);
    {
      const auto expected = c2h::make_host_buffer<int>(stream, device, {0, 2, 9, 5, 8});
      REQUIRE_THAT(d_unique_out, c2h::Equals(stream, expected));
    }
    {
      const auto expected = c2h::make_host_buffer<int>(stream, device, {0, 1, 6, 2, 4});
      REQUIRE_THAT(d_aggregates_out, c2h::Equals(stream, expected));
    }
  });
}

#endif // TEST_LAUNCH == 0

// The graph launch appends its own stream argument, which cannot be combined with an environment
#if TEST_LAUNCH != 2

CUB_TEST_CASE("Device Reduce two-phase accepts legacy null streams", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_in = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9});
  auto d_out      = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);
  const stream_scope scope{cudaStream_t{}};

  SECTION("nullptr")
  {
    stream.sync();
    device_reduce_two_phase(d_in.data(), d_out.data(), 7, cuda::std::plus<>{}, 10, nullptr);
  }
  SECTION("literal 0")
  {
    stream.sync();
    device_reduce_two_phase(d_in.data(), d_out.data(), 7, cuda::std::plus<>{}, 10, 0);
  }
  SECTION("cudaStream_t{}")
  {
    stream.sync();
    device_reduce_two_phase(d_in.data(), d_out.data(), 7, cuda::std::plus<>{}, 10, cudaStream_t{});
  }

  REQUIRE(c2h::read_single(stream, d_out) == 48);
}

CUB_TEST_CASE("Device Sum two-phase accepts legacy null streams", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_in = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9});
  auto d_out      = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);
  const stream_scope scope{cudaStream_t{}};

  SECTION("nullptr")
  {
    stream.sync();
    device_reduce_sum_two_phase(d_in.data(), d_out.data(), 7, nullptr);
  }
  SECTION("literal 0")
  {
    stream.sync();
    device_reduce_sum_two_phase(d_in.data(), d_out.data(), 7, 0);
  }
  SECTION("cudaStream_t{}")
  {
    stream.sync();
    device_reduce_sum_two_phase(d_in.data(), d_out.data(), 7, cudaStream_t{});
  }

  REQUIRE(c2h::read_single(stream, d_out) == 38);
}

CUB_TEST_CASE("Device Min two-phase accepts legacy null streams", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_in = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9});
  auto d_out      = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);
  const stream_scope scope{cudaStream_t{}};

  SECTION("nullptr")
  {
    stream.sync();
    device_reduce_min_two_phase(d_in.data(), d_out.data(), 7, nullptr);
  }
  SECTION("literal 0")
  {
    stream.sync();
    device_reduce_min_two_phase(d_in.data(), d_out.data(), 7, 0);
  }
  SECTION("cudaStream_t{}")
  {
    stream.sync();
    device_reduce_min_two_phase(d_in.data(), d_out.data(), 7, cudaStream_t{});
  }

  REQUIRE(c2h::read_single(stream, d_out) == 0);
}

CUB_TEST_CASE("Device Max two-phase accepts legacy null streams", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_in = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9});
  auto d_out      = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);
  const stream_scope scope{cudaStream_t{}};

  SECTION("nullptr")
  {
    stream.sync();
    device_reduce_max_two_phase(d_in.data(), d_out.data(), 7, nullptr);
  }
  SECTION("literal 0")
  {
    stream.sync();
    device_reduce_max_two_phase(d_in.data(), d_out.data(), 7, 0);
  }
  SECTION("cudaStream_t{}")
  {
    stream.sync();
    device_reduce_max_two_phase(d_in.data(), d_out.data(), 7, cudaStream_t{});
  }

  REQUIRE(c2h::read_single(stream, d_out) == 9);
}

CUB_TEST_CASE("Device TransformReduce two-phase accepts legacy null streams", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_in = c2h::make_device_buffer<int>(stream, device, {8, 6, 7, 5, 3, 0, 9});
  auto d_out      = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);
  const stream_scope scope{cudaStream_t{}};

  SECTION("nullptr")
  {
    stream.sync();
    device_transform_reduce_two_phase(
      d_in.data(), d_out.data(), 7, cuda::std::plus<>{}, cuda::std::negate<int>{}, 0, nullptr);
  }
  SECTION("literal 0")
  {
    stream.sync();
    device_transform_reduce_two_phase(d_in.data(), d_out.data(), 7, cuda::std::plus<>{}, cuda::std::negate<int>{}, 0, 0);
  }
  SECTION("cudaStream_t{}")
  {
    stream.sync();
    device_transform_reduce_two_phase(
      d_in.data(), d_out.data(), 7, cuda::std::plus<>{}, cuda::std::negate<int>{}, 0, cudaStream_t{});
  }

  REQUIRE(c2h::read_single(stream, d_out) == -38);
}

CUB_TEST_CASE("Device ReduceByKey two-phase accepts legacy null streams", "[reduce][device]", CUB_SMALL)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  const auto d_keys_in   = c2h::make_device_buffer<int>(stream, device, {0, 2, 2, 9, 5, 5, 5, 8});
  const auto d_values_in = c2h::make_device_buffer<int>(stream, device, {0, 7, 1, 6, 2, 5, 3, 4});
  auto d_unique_out      = c2h::make_device_buffer<int>(stream, device, 5, cuda::no_init);
  auto d_aggregates_out  = c2h::make_device_buffer<int>(stream, device, 5, cuda::no_init);
  auto d_num_runs_out    = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);
  const stream_scope scope{cudaStream_t{}};

  SECTION("nullptr")
  {
    stream.sync();
    device_reduce_by_key_two_phase(
      d_keys_in.data(),
      d_unique_out.data(),
      d_values_in.data(),
      d_aggregates_out.data(),
      d_num_runs_out.data(),
      cuda::minimum<>{},
      8,
      nullptr);
  }
  SECTION("literal 0")
  {
    stream.sync();
    device_reduce_by_key_two_phase(
      d_keys_in.data(),
      d_unique_out.data(),
      d_values_in.data(),
      d_aggregates_out.data(),
      d_num_runs_out.data(),
      cuda::minimum<>{},
      8,
      0);
  }
  SECTION("cudaStream_t{}")
  {
    stream.sync();
    device_reduce_by_key_two_phase(
      d_keys_in.data(),
      d_unique_out.data(),
      d_values_in.data(),
      d_aggregates_out.data(),
      d_num_runs_out.data(),
      cuda::minimum<>{},
      8,
      cudaStream_t{});
  }

  REQUIRE(c2h::read_single(stream, d_num_runs_out) == 5);
  {
    const auto expected = c2h::make_host_buffer<int>(stream, device, {0, 2, 9, 5, 8});
    REQUIRE_THAT(d_unique_out, c2h::Equals(stream, expected));
  }
  {
    const auto expected = c2h::make_host_buffer<int>(stream, device, {0, 1, 6, 2, 4});
    REQUIRE_THAT(d_aggregates_out, c2h::Equals(stream, expected));
  }
}

// The tuning environments are empty classes, which older CTKs (at least 12.9 and 13.0) cannot pass as kernel
// parameters, so these tests cannot use the device launch (TEST_LAUNCH == 1).
#  if TEST_LAUNCH == 0

CUB_TEST("Device Reduce two-phase can be tuned", "[reduce][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const auto d_in                          = block_size_extracting_constant_iterator(1, d_block_size.data());
  auto d_out                               = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);
  const auto env                           = cuda::execution::tune(reduce_tuning<target_block_size>{});

  stream.sync();
  device_reduce_two_phase(d_in, d_out.data(), 1, cuda::std::plus<>{}, 0, env);

  REQUIRE(c2h::read_single(stream, d_out) == 1);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device Sum two-phase can be tuned", "[reduce][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const auto d_in                          = block_size_extracting_constant_iterator(1, d_block_size.data());
  auto d_out                               = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);
  const auto env                           = cuda::execution::tune(reduce_tuning<target_block_size>{});

  stream.sync();
  device_reduce_sum_two_phase(d_in, d_out.data(), 1, env);

  REQUIRE(c2h::read_single(stream, d_out) == 1);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device Min two-phase can be tuned", "[reduce][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const auto d_in                          = block_size_extracting_constant_iterator(42, d_block_size.data());
  auto d_out                               = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);
  const auto env                           = cuda::execution::tune(reduce_tuning<target_block_size>{});

  stream.sync();
  device_reduce_min_two_phase(d_in, d_out.data(), 1, env);

  REQUIRE(c2h::read_single(stream, d_out) == 42);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device Max two-phase can be tuned", "[reduce][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const auto d_in                          = block_size_extracting_constant_iterator(42, d_block_size.data());
  auto d_out                               = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);
  const auto env                           = cuda::execution::tune(reduce_tuning<target_block_size>{});

  stream.sync();
  device_reduce_max_two_phase(d_in, d_out.data(), 1, env);

  REQUIRE(c2h::read_single(stream, d_out) == 42);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device TransformReduce two-phase can be tuned", "[reduce][device]", CUB_SMALL, block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const auto d_in                          = block_size_extracting_constant_iterator(1, d_block_size.data());
  auto d_out                               = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);
  const auto env                           = cuda::execution::tune(reduce_tuning<target_block_size>{});

  stream.sync();
  device_transform_reduce_two_phase(d_in, d_out.data(), 1, cuda::std::plus<>{}, cuda::std::negate<int>{}, 0, env);

  REQUIRE(c2h::read_single(stream, d_out) == -1);
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

CUB_TEST("Device ReduceByKey two-phase can be tuned", "[reduce][device]", CUB_SMALL, reduce_by_key_block_sizes)
{
  const auto stream = c2h::default_stream_ref();
  const auto device = c2h::current_device();

  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  const auto d_keys_in                     = c2h::make_device_buffer<int>(stream, device, {0, 2, 2, 9, 5, 5, 5, 8});
  const auto d_values_in                   = c2h::make_device_buffer<int>(stream, device, {0, 7, 1, 6, 2, 5, 3, 4});
  auto d_unique_out                        = c2h::make_device_buffer<int>(stream, device, 5, cuda::no_init);
  auto d_aggregates_out                    = c2h::make_device_buffer<int>(stream, device, 5, cuda::no_init);
  auto d_num_runs_out                      = c2h::make_device_buffer<int>(stream, device, 1, cuda::no_init);
  auto d_block_size                        = c2h::make_device_buffer<unsigned int>(stream, device, {0u});
  const block_size_extracting_op<cuda::minimum<int>> reduction_op{d_block_size.data()};
  const auto env = cuda::execution::tune(reduce_by_key_tuning<target_block_size>{});

  stream.sync();
  device_reduce_by_key_two_phase(
    d_keys_in.data(),
    d_unique_out.data(),
    d_values_in.data(),
    d_aggregates_out.data(),
    d_num_runs_out.data(),
    reduction_op,
    8,
    env);

  REQUIRE(c2h::read_single(stream, d_num_runs_out) == 5);
  {
    const auto expected = c2h::make_host_buffer<int>(stream, device, {0, 2, 9, 5, 8});
    REQUIRE_THAT(d_unique_out, c2h::Equals(stream, expected));
  }
  {
    const auto expected = c2h::make_host_buffer<int>(stream, device, {0, 1, 6, 2, 4});
    REQUIRE_THAT(d_aggregates_out, c2h::Equals(stream, expected));
  }
  REQUIRE(c2h::read_single(stream, d_block_size) == target_block_size);
}

#  endif // TEST_LAUNCH == 0

#endif // TEST_LAUNCH != 2

#if _CCCL_COMPILER(GCC, >=, 8) // gcc 7 cannot preserve constexpr-ness from p1 to p2
CUB_TEST("Test ReducePolicy properties", "[reduce][device]", CUB_SMALL)
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::ReducePolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::ReducePolicy>);

  STATIC_REQUIRE(::cuda::std::semiregular<cub::ReducePassPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::ReducePassPolicy>);

  // aggregate init
  constexpr auto p1_multi = cub::ReducePassPolicy{
    256, 16, 4, cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS, cub::CacheLoadModifier::LOAD_LDG};
  constexpr auto p1_single = cub::ReducePassPolicy{
    128, 8, 2, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, cub::CacheLoadModifier::LOAD_DEFAULT};
  constexpr auto p1 = cub::ReducePolicy{p1_multi, p1_single};

#  if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto p2_multi = cub::ReducePassPolicy{
    .threads_per_block = 256,
    .items_per_thread  = 16,
    .vec_size          = 4,
    .reduce_algorithm  = cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS,
    .load_modifier     = cub::CacheLoadModifier::LOAD_LDG};
  constexpr auto p2_single = cub::ReducePassPolicy{
    .threads_per_block = 128,
    .items_per_thread  = 8,
    .vec_size          = 2,
    .reduce_algorithm  = cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
    .load_modifier     = cub::CacheLoadModifier::LOAD_DEFAULT};
  constexpr auto p2 = cub::ReducePolicy{.multi_tile = p2_multi, .single_tile = p2_single};
#  else // _CCCL_STD_VER >= 2020
  constexpr auto p2_multi  = p1_multi;
  constexpr auto p2_single = p1_single;
  constexpr auto p2        = p1;
#  endif // _CCCL_STD_VER >= 2020

  // comparison
  STATIC_REQUIRE(p1_multi == p2_multi);
  STATIC_REQUIRE_FALSE(p1_multi != p2_multi);

  STATIC_REQUIRE(p1_single == p2_single);
  STATIC_REQUIRE_FALSE(p1_single != p2_single);

  STATIC_REQUIRE(p1 == p2);
  STATIC_REQUIRE_FALSE(p1 != p2);

  auto to_string = [](const auto& p) {
    std::ostringstream os;
    os << p;
    return os.str();
  };
  REQUIRE(to_string(p1_multi)
          == "ReducePassPolicy { .threads_per_block = 256, .items_per_thread = 16, .vec_size = 4"
             ", .reduce_algorithm = BLOCK_REDUCE_WARP_REDUCTIONS, .load_modifier = LOAD_LDG }");
  REQUIRE(to_string(p1_single)
          == "ReducePassPolicy { .threads_per_block = 128, .items_per_thread = 8, .vec_size = 2"
             ", .reduce_algorithm = BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, .load_modifier = LOAD_DEFAULT }");
  REQUIRE(to_string(p1)
          == "ReducePolicy { .multi_tile = ReducePassPolicy { .threads_per_block = 256"
             ", .items_per_thread = 16, .vec_size = 4"
             ", .reduce_algorithm = BLOCK_REDUCE_WARP_REDUCTIONS, .load_modifier = LOAD_LDG }"
             ", .single_tile = ReducePassPolicy { .threads_per_block = 128"
             ", .items_per_thread = 8, .vec_size = 2"
             ", .reduce_algorithm = BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, .load_modifier = LOAD_DEFAULT } }");
}

CUB_TEST("Test ReduceByKeyPolicy properties", "[reduce][device]", CUB_SMALL)
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::ReduceByKeyPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::ReduceByKeyPolicy>);

  // aggregate init
  constexpr auto p1 = cub::ReduceByKeyPolicy{
    cub::ReduceByKeyAlgorithm::lookback,
    {128,
     7,
     cub::BLOCK_LOAD_DIRECT,
     cub::LOAD_DEFAULT,
     cub::BLOCK_SCAN_WARP_SCANS,
     {cub::LookbackDelayAlgorithm::fixed_delay, 832, 1165}}};

#  if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto p2 = cub::ReduceByKeyPolicy{
    .algorithm = cub::ReduceByKeyAlgorithm::lookback,
    .lookback  = cub::ReduceByKeyLookbackPolicy{
      .threads_per_block = 128,
      .items_per_thread  = 7,
      .load_algorithm    = cub::BLOCK_LOAD_DIRECT,
      .load_modifier     = cub::LOAD_DEFAULT,
      .scan_algorithm    = cub::BLOCK_SCAN_WARP_SCANS,
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
          == "ReduceByKeyPolicy { .algorithm = ReduceByKeyAlgorithm::lookback"
             ", .lookback = ReduceByKeyLookbackPolicy { .threads_per_block = 128, .items_per_thread = 7"
             ", .load_algorithm = BLOCK_LOAD_DIRECT, .load_modifier = LOAD_DEFAULT"
             ", .scan_algorithm = BLOCK_SCAN_WARP_SCANS"
             ", .lookback_delay = LookbackDelayPolicy { .kind = LookbackDelayAlgorithm::fixed_delay"
             ", .delay = 832, .l2_write_latency = 1165 } } }");
}
#endif // _CCCL_COMPILER(GCC, >=, 8)
