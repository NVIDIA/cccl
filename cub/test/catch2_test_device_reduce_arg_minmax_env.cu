// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/execution>
#include <cuda/iterator>
#include <cuda/std/execution>
#include <cuda/std/utility>
#include <cuda/stream>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMinMax, device_arg_minmax);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMinLastMax, device_arg_minlastmax);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include "cub_test_macros.h"

namespace stdexec = cuda::std::execution;

template <typename... Args>
cudaError_t call_argminmax_api(bool last_max, Args&&... args)
{
  return last_max ? cub::DeviceReduce::ArgMinLastMax(::cuda::std::forward<Args>(args)...)
                  : cub::DeviceReduce::ArgMinMax(::cuda::std::forward<Args>(args)...);
}

template <typename... Args>
void call_argminmax_launch_wrapper(bool last_max, Args&&... args)
{
  if (last_max)
  {
    device_arg_minlastmax(::cuda::std::forward<Args>(args)...);
  }
  else
  {
    device_arg_minmax(::cuda::std::forward<Args>(args)...);
  }
}

// Custom tuning that forces a specific block size, used to verify a tuning environment reaches the kernel.
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

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 32>, cuda::std::integral_constant<unsigned int, 64>>;

#if TEST_LAUNCH == 0

CUB_TEST_CASE("Device ArgMin[Last]Max works with default environment", "[reduce][device]", CUB_SMALL)
{
  const bool last_max = GENERATE(false, true);

  auto input     = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_out   = c2h::device_vector<float>(1, thrust::no_init);
  auto min_index = c2h::device_vector<cuda::std::int64_t>(1, thrust::no_init);
  auto max_out   = c2h::device_vector<float>(1, thrust::no_init);
  auto max_index = c2h::device_vector<cuda::std::int64_t>(1, thrust::no_init);

  auto error = call_argminmax_api(
    last_max,
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    static_cast<::cuda::std::int64_t>(input.size()));
  REQUIRE(error == cudaSuccess);
  REQUIRE(min_out[0] == 0.0f);
  REQUIRE(min_index[0] == 3);
  REQUIRE(max_out[0] == 4.0f);
  REQUIRE(max_index[0] == 2);
}

#endif // TEST_LAUNCH == 0

CUB_TEST("Device ArgMin[Last]Max uses environment", "[reduce][device]", CUB_SMALL)
{
  const bool last_max = GENERATE(false, true);

  auto input     = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_out   = c2h::device_vector<float>(1, thrust::no_init);
  auto min_index = c2h::device_vector<cuda::std::int64_t>(1, thrust::no_init);
  auto max_out   = c2h::device_vector<float>(1, thrust::no_init);
  auto max_index = c2h::device_vector<cuda::std::int64_t>(1, thrust::no_init);

  const auto n = static_cast<::cuda::std::int64_t>(input.size());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == call_argminmax_api(
      last_max,
      nullptr,
      expected_bytes_allocated,
      input.begin(),
      min_out.begin(),
      min_index.begin(),
      max_out.begin(),
      max_index.begin(),
      n));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  call_argminmax_launch_wrapper(
    last_max, input.begin(), min_out.begin(), min_index.begin(), max_out.begin(), max_index.begin(), n, env);

  REQUIRE(min_out[0] == 0.0f);
  REQUIRE(min_index[0] == 3);
  REQUIRE(max_out[0] == 4.0f);
  REQUIRE(max_index[0] == 2);
}

CUB_TEST("Device ArgMin[Last]Max with compare_op uses environment", "[reduce][device]", CUB_SMALL)
{
  const bool last_max = GENERATE(false, true);

  auto input     = c2h::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_out   = c2h::device_vector<float>(1, thrust::no_init);
  auto min_index = c2h::device_vector<cuda::std::int64_t>(1, thrust::no_init);
  auto max_out   = c2h::device_vector<float>(1, thrust::no_init);
  auto max_index = c2h::device_vector<cuda::std::int64_t>(1, thrust::no_init);

  const auto n = static_cast<::cuda::std::int64_t>(input.size());

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == call_argminmax_api(
      last_max,
      nullptr,
      expected_bytes_allocated,
      input.begin(),
      min_out.begin(),
      min_index.begin(),
      max_out.begin(),
      max_index.begin(),
      n,
      cuda::std::less{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  call_argminmax_launch_wrapper(
    last_max,
    input.begin(),
    min_out.begin(),
    min_index.begin(),
    max_out.begin(),
    max_index.begin(),
    n,
    cuda::std::less{},
    env);

  REQUIRE(min_out[0] == 0.0f);
  REQUIRE(min_index[0] == 3);
  REQUIRE(max_out[0] == 4.0f);
  REQUIRE(max_index[0] == 2);
}

#if TEST_LAUNCH == 0
CUB_TEST("Device ArgMin[Last]Max uses custom stream", "[reduce][device]", CUB_SMALL)
{
  const bool last_max = GENERATE(false, true);

  // The maximum value 4 appears twice, so first-max and last-max disagree on the reported index.
  auto input     = c2h::device_vector<float>{3.0f, 4.0f, 4.0f, 0.0f, 2.0f};
  auto min_out   = c2h::device_vector<float>(1, thrust::no_init);
  auto min_index = c2h::device_vector<cuda::std::int64_t>(1, thrust::no_init);
  auto max_out   = c2h::device_vector<float>(1, thrust::no_init);
  auto max_index = c2h::device_vector<cuda::std::int64_t>(1, thrust::no_init);

  const auto n                                = static_cast<::cuda::std::int64_t>(input.size());
  const cuda::std::int64_t expected_max_index = last_max ? 2 : 1;

  cuda::stream stream{cuda::devices[0]};
  const auto error = call_argminmax_api(
    last_max, input.begin(), min_out.begin(), min_index.begin(), max_out.begin(), max_index.begin(), n, stream);
  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(min_out[0] == 0.0f);
  REQUIRE(min_index[0] == 3);
  REQUIRE(max_out[0] == 4.0f);
  REQUIRE(max_index[0] == expected_max_index);
}
#endif // TEST_LAUNCH == 0

// Device-side (CDP) launch cannot apply a host-provided tuning environment, so skip it there (matches the other
// tuning tests in catch2_test_device_reduce_env.cu).
#if TEST_LAUNCH != 1
CUB_TEST("Device ArgMin[Last]Max can be tuned", "[reduce][device]", CUB_SMALL, block_sizes)
{
  const bool last_max                      = GENERATE(false, true);
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  c2h::device_vector<unsigned int> d_block_size(1);
  using compare_t = block_size_extracting_op<cuda::std::less<>>;
  compare_t compare_op{thrust::raw_pointer_cast(d_block_size.data())};

  // The maximum value 4 appears twice, so first-max and last-max disagree on the reported index.
  auto input     = c2h::device_vector<int>{3, 4, 4, 0, 2};
  auto min_out   = c2h::device_vector<int>(1);
  auto min_index = c2h::device_vector<cuda::std::int64_t>(1);
  auto max_out   = c2h::device_vector<int>(1);
  auto max_index = c2h::device_vector<cuda::std::int64_t>(1);

  const auto n                                = static_cast<int>(input.size());
  const cuda::std::int64_t expected_max_index = last_max ? 2 : 1;

  auto env = cuda::execution::tune(reduce_tuning<target_block_size>{});

  call_argminmax_launch_wrapper(
    last_max, input.begin(), min_out.begin(), min_index.begin(), max_out.begin(), max_index.begin(), n, compare_op, env);

  REQUIRE(min_out[0] == 0);
  REQUIRE(min_index[0] == 3);
  REQUIRE(max_out[0] == 4);
  REQUIRE(max_index[0] == expected_max_index);
  REQUIRE(d_block_size[0] == target_block_size);
}
#endif // TEST_LAUNCH != 1
