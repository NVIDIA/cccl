// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_merge.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/__execution/policy.h>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <sstream>

#include "block_size_extracting_helpers.h"
#include "catch2_test_launch_helper.h"
#include <c2h/device_and_stream.h>

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceMerge::MergeKeys, merge_keys);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceMerge::MergePairs, merge_pairs);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <cuda/__execution/require.h>

#include "cub_test_macros.h"

namespace stdexec = cuda::std::execution;

using block_size_extracting_less_t = block_size_extracting_op<cuda::std::less<>>;

template <int ThreadsPerBlock>
struct merge_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::MergePolicy
  {
    return {ThreadsPerBlock, 1, cub::LOAD_DEFAULT, cub::BLOCK_STORE_WARP_TRANSPOSE, false, false};
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

#if TEST_LAUNCH == 0

CUB_TEST_CASE("DeviceMerge::MergeKeys works with default environment", "[merge][device]", CUB_SMALL)
{
  auto keys1  = c2h::device_vector<int>{0, 2, 5};
  auto keys2  = c2h::device_vector<int>{0, 3, 3, 4};
  auto result = c2h::device_vector<int>(7);

  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergeKeys(
      keys1.begin(), static_cast<int>(keys1.size()), keys2.begin(), static_cast<int>(keys2.size()), result.begin()));

  const c2h::device_vector<int> expected{0, 0, 2, 3, 3, 4, 5};
  REQUIRE(result == expected);
}

CUB_TEST_CASE("DeviceMerge::MergePairs works with default environment", "[merge][device]", CUB_SMALL)
{
  auto keys1   = c2h::device_vector<int>{0, 2, 5};
  auto values1 = c2h::device_vector<char>{'a', 'b', 'c'};
  auto keys2   = c2h::device_vector<int>{0, 3, 3, 4};
  auto values2 = c2h::device_vector<char>{'A', 'B', 'C', 'D'};

  auto result_keys   = c2h::device_vector<int>(7);
  auto result_values = c2h::device_vector<char>(7);

  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergePairs(
      keys1.begin(),
      values1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      values2.begin(),
      static_cast<int>(keys2.size()),
      result_keys.begin(),
      result_values.begin()));

  const c2h::device_vector<int> expected_keys{0, 0, 2, 3, 3, 4, 5};
  const c2h::device_vector<char> expected_values{'a', 'A', 'b', 'B', 'C', 'D', 'c'};
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

#endif

CUB_TEST("DeviceMerge::MergeKeys can be tuned", "[merge][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys1                               = c2h::device_vector<int>{0, 2, 5};
  auto keys2                               = c2h::device_vector<int>{0, 3, 3, 4};
  auto result                              = c2h::device_vector<int>(7);
  auto d_block_size                        = c2h::device_vector<unsigned int>(1);

  const block_size_extracting_less_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  auto env = cuda::execution::tune(merge_tuning<target_block_size>{});

  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergeKeys(
      keys1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      static_cast<int>(keys2.size()),
      result.begin(),
      block_size_check,
      env));

  const c2h::device_vector<int> expected{0, 0, 2, 3, 3, 4, 5};
  REQUIRE(result == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceMerge::MergePairs can be tuned", "[merge][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys1                               = c2h::device_vector<int>{0, 2, 5};
  auto values1                             = c2h::device_vector<char>{'a', 'b', 'c'};
  auto keys2                               = c2h::device_vector<int>{0, 3, 3, 4};
  auto values2                             = c2h::device_vector<char>{'A', 'B', 'C', 'D'};
  auto result_keys                         = c2h::device_vector<int>(7);
  auto result_values                       = c2h::device_vector<char>(7);
  auto d_block_size                        = c2h::device_vector<unsigned int>(1);

  const block_size_extracting_less_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  auto env = cuda::execution::tune(merge_tuning<target_block_size>{});

  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergePairs(
      keys1.begin(),
      values1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      values2.begin(),
      static_cast<int>(keys2.size()),
      result_keys.begin(),
      result_values.begin(),
      block_size_check,
      env));

  const c2h::device_vector<int> expected_keys{0, 0, 2, 3, 3, 4, 5};
  const c2h::device_vector<char> expected_values{'a', 'A', 'b', 'B', 'C', 'D', 'c'};
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceMerge::MergeKeys uses environment", "[merge][device]", CUB_SMALL)
{
  auto keys1  = c2h::device_vector<int>{0, 2, 5};
  auto keys2  = c2h::device_vector<int>{0, 3, 3, 4};
  auto result = c2h::device_vector<int>(7);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergeKeys(
      nullptr,
      expected_bytes_allocated,
      keys1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      static_cast<int>(keys2.size()),
      result.begin()));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  merge_keys(keys1.begin(),
             static_cast<int>(keys1.size()),
             keys2.begin(),
             static_cast<int>(keys2.size()),
             result.begin(),
             cuda::std::less<>{},
             env);

  const c2h::device_vector<int> expected{0, 0, 2, 3, 3, 4, 5};
  REQUIRE(result == expected);
}

CUB_TEST_CASE("DeviceMerge::MergeKeys uses custom stream", "[merge][device]", CUB_SMALL)
{
  auto keys1  = c2h::device_vector<int>{0, 2, 5};
  auto keys2  = c2h::device_vector<int>{0, 3, 3, 4};
  auto result = c2h::device_vector<int>(7);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergeKeys(
      nullptr,
      expected_bytes_allocated,
      keys1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      static_cast<int>(keys2.size()),
      result.begin()));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  merge_keys(keys1.begin(),
             static_cast<int>(keys1.size()),
             keys2.begin(),
             static_cast<int>(keys2.size()),
             result.begin(),
             cuda::std::less<>{},
             env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  const c2h::device_vector<int> expected{0, 0, 2, 3, 3, 4, 5};
  REQUIRE(result == expected);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

CUB_TEST("DeviceMerge::MergePairs uses environment", "[merge][device]", CUB_SMALL)
{
  auto keys1   = c2h::device_vector<int>{0, 2, 5};
  auto values1 = c2h::device_vector<char>{'a', 'b', 'c'};
  auto keys2   = c2h::device_vector<int>{0, 3, 3, 4};
  auto values2 = c2h::device_vector<char>{'A', 'B', 'C', 'D'};

  auto result_keys   = c2h::device_vector<int>(7);
  auto result_values = c2h::device_vector<char>(7);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergePairs(
      nullptr,
      expected_bytes_allocated,
      keys1.begin(),
      values1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      values2.begin(),
      static_cast<int>(keys2.size()),
      result_keys.begin(),
      result_values.begin()));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  merge_pairs(
    keys1.begin(),
    values1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    values2.begin(),
    static_cast<int>(keys2.size()),
    result_keys.begin(),
    result_values.begin(),
    cuda::std::less<>{},
    env);

  const c2h::device_vector<int> expected_keys{0, 0, 2, 3, 3, 4, 5};
  const c2h::device_vector<char> expected_values{'a', 'A', 'b', 'B', 'C', 'D', 'c'};
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

CUB_TEST_CASE("DeviceMerge::MergePairs uses custom stream", "[merge][device]", CUB_SMALL)
{
  auto keys1   = c2h::device_vector<int>{0, 2, 5};
  auto values1 = c2h::device_vector<char>{'a', 'b', 'c'};
  auto keys2   = c2h::device_vector<int>{0, 3, 3, 4};
  auto values2 = c2h::device_vector<char>{'A', 'B', 'C', 'D'};

  auto result_keys   = c2h::device_vector<int>(7);
  auto result_values = c2h::device_vector<char>(7);

  cudaStream_t custom_stream;
  REQUIRE(cudaSuccess == cudaStreamCreate(&custom_stream));

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergePairs(
      nullptr,
      expected_bytes_allocated,
      keys1.begin(),
      values1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      values2.begin(),
      static_cast<int>(keys2.size()),
      result_keys.begin(),
      result_values.begin()));

  auto stream_prop = stdexec::prop{cuda::get_stream_t{}, cuda::stream_ref{custom_stream}};
  auto env         = stdexec::env{stream_prop, expected_allocation_size(expected_bytes_allocated)};

  merge_pairs(
    keys1.begin(),
    values1.begin(),
    static_cast<int>(keys1.size()),
    keys2.begin(),
    values2.begin(),
    static_cast<int>(keys2.size()),
    result_keys.begin(),
    result_values.begin(),
    cuda::std::less<>{},
    env);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(custom_stream));

  const c2h::device_vector<int> expected_keys{0, 0, 2, 3, 3, 4, 5};
  const c2h::device_vector<char> expected_values{'a', 'A', 'b', 'B', 'C', 'D', 'c'};
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);

  REQUIRE(cudaSuccess == cudaStreamDestroy(custom_stream));
}

struct no_unroll_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::MergePolicy
  {
    return {256, 7, cub::LOAD_DEFAULT, cub::BLOCK_STORE_WARP_TRANSPOSE, false, false, false};
  }
};

CUB_TEST_CASE("DeviceMerge::MergeKeys works with unroll disabled", "[merge][device]", CUB_SMALL)
{
  auto keys1  = c2h::device_vector<int>{0, 2, 5};
  auto keys2  = c2h::device_vector<int>{0, 3, 3, 4};
  auto result = c2h::device_vector<int>(7);
  auto env    = cuda::execution::tune(no_unroll_tuning{});

  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergeKeys(
      keys1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      static_cast<int>(keys2.size()),
      result.begin(),
      cuda::std::less<>{},
      env));

  const c2h::device_vector<int> expected{0, 0, 2, 3, 3, 4, 5};
  REQUIRE(result == expected);
}

CUB_TEST_CASE("DeviceMerge::MergePairs works with unroll disabled", "[merge][device]", CUB_SMALL)
{
  auto keys1   = c2h::device_vector<int>{0, 2, 5};
  auto values1 = c2h::device_vector<char>{'a', 'b', 'c'};
  auto keys2   = c2h::device_vector<int>{0, 3, 3, 4};
  auto values2 = c2h::device_vector<char>{'A', 'B', 'C', 'D'};

  auto result_keys   = c2h::device_vector<int>(7);
  auto result_values = c2h::device_vector<char>(7);
  auto env           = cuda::execution::tune(no_unroll_tuning{});

  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergePairs(
      keys1.begin(),
      values1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      values2.begin(),
      static_cast<int>(keys2.size()),
      result_keys.begin(),
      result_values.begin(),
      cuda::std::less<>{},
      env));

  const c2h::device_vector<int> expected_keys{0, 0, 2, 3, 3, 4, 5};
  const c2h::device_vector<char> expected_values{'a', 'A', 'b', 'B', 'C', 'D', 'c'};
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

#if TEST_LAUNCH == 0

// The two-phase overloads take the same environment as the single-phase ones but never allocate, so they do not go
// through the launch wrappers and would run identically in every TEST_LAUNCH variant. Test them with host launch only.

// Runs the two-phase overload wrapped by two_phase once per supported kind of environment: queries the temporary
// storage size, executes with user provided storage, and checks that every kernel is launched on the stream carried
// by the environment (or on the default stream if the environment carries none).
template <class TwoPhaseFn>
void test_two_phase_env_kinds(size_t expected_temp_storage_bytes, TwoPhaseFn two_phase)
{
  const cuda::stream stream = c2h::make_current_device_stream();
  const cuda::stream_ref default_stream{cudaStream_t{}};

  auto run = [&](const auto& env, cuda::stream_ref expected_stream) {
    size_t temp_storage_bytes = 0;
    REQUIRE(cudaSuccess == two_phase(nullptr, temp_storage_bytes, env));
    REQUIRE(temp_storage_bytes == expected_temp_storage_bytes);

    c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
    {
      // stream_registry_factory_t fails the test if a kernel is launched on any other stream
      const stream_scope scope{expected_stream.get()};
      REQUIRE(cudaSuccess == two_phase(thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, env));
    }
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    expected_stream.sync();
  };

  SECTION("default environment")
  {
    run(stdexec::env<>{}, default_stream);
  }

  SECTION("cudaStream_t")
  {
    run(stream.get(), cuda::stream_ref{stream});
  }

  SECTION("cuda::stream")
  {
    run(stream, cuda::stream_ref{stream});
  }

  SECTION("cuda::stream_ref")
  {
    run(cuda::stream_ref{stream}, cuda::stream_ref{stream});
  }

  SECTION("environment with stream")
  {
    run(stdexec::env{cuda::stream_ref{stream}}, cuda::stream_ref{stream});
  }

  SECTION("cuda::execution::gpu")
  {
    run(cuda::execution::gpu, default_stream);
  }

  SECTION("cuda::execution::gpu with stream")
  {
    run(cuda::execution::gpu.with(cuda::get_stream, cuda::stream_ref{stream}), cuda::stream_ref{stream});
  }
}

CUB_TEST_CASE("DeviceMerge::MergeKeys works with user provided memory and environment", "[merge][device]", CUB_SMALL)
{
  auto keys1  = c2h::device_vector<int>{0, 2, 5};
  auto keys2  = c2h::device_vector<int>{0, 3, 3, 4};
  auto result = c2h::device_vector<int>(7);

  size_t expected_bytes{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergeKeys(
      nullptr,
      expected_bytes,
      keys1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      static_cast<int>(keys2.size()),
      result.begin()));

  test_two_phase_env_kinds(expected_bytes, [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
    return cub::DeviceMerge::MergeKeys(
      d_temp_storage,
      temp_storage_bytes,
      keys1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      static_cast<int>(keys2.size()),
      result.begin(),
      cuda::std::less<>{},
      env);
  });

  const c2h::device_vector<int> expected{0, 0, 2, 3, 3, 4, 5};
  REQUIRE(result == expected);
}

CUB_TEST_CASE("DeviceMerge::MergePairs works with user provided memory and environment", "[merge][device]", CUB_SMALL)
{
  auto keys1   = c2h::device_vector<int>{0, 2, 5};
  auto values1 = c2h::device_vector<char>{'a', 'b', 'c'};
  auto keys2   = c2h::device_vector<int>{0, 3, 3, 4};
  auto values2 = c2h::device_vector<char>{'A', 'B', 'C', 'D'};

  auto result_keys   = c2h::device_vector<int>(7);
  auto result_values = c2h::device_vector<char>(7);

  size_t expected_bytes{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceMerge::MergePairs(
      nullptr,
      expected_bytes,
      keys1.begin(),
      values1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      values2.begin(),
      static_cast<int>(keys2.size()),
      result_keys.begin(),
      result_values.begin()));

  test_two_phase_env_kinds(expected_bytes, [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
    return cub::DeviceMerge::MergePairs(
      d_temp_storage,
      temp_storage_bytes,
      keys1.begin(),
      values1.begin(),
      static_cast<int>(keys1.size()),
      keys2.begin(),
      values2.begin(),
      static_cast<int>(keys2.size()),
      result_keys.begin(),
      result_values.begin(),
      cuda::std::less<>{},
      env);
  });

  const c2h::device_vector<int> expected_keys{0, 0, 2, 3, 3, 4, 5};
  const c2h::device_vector<char> expected_values{'a', 'A', 'b', 'B', 'C', 'D', 'c'};
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
}

// Before the environment parameter, the two-phase overloads took `cudaStream_t stream = nullptr`, so callers passing
// nullptr or a literal 0 for the stream exist. Both must keep compiling and keep running on the default stream.
CUB_TEST_CASE("DeviceMerge two-phase overloads accept legacy null stream arguments", "[merge][device]", CUB_SMALL)
{
  auto keys1  = c2h::device_vector<int>{0, 2, 5};
  auto keys2  = c2h::device_vector<int>{0, 3, 3, 4};
  auto result = c2h::device_vector<int>(7);

  auto merge_keys_on = [&](const auto& stream) {
    size_t temp_storage_bytes = 0;
    REQUIRE(
      cudaSuccess
      == cub::DeviceMerge::MergeKeys(
        nullptr,
        temp_storage_bytes,
        keys1.begin(),
        static_cast<int>(keys1.size()),
        keys2.begin(),
        static_cast<int>(keys2.size()),
        result.begin(),
        cuda::std::less<>{},
        stream));

    c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
    const stream_scope scope{cudaStream_t{}};
    REQUIRE(
      cudaSuccess
      == cub::DeviceMerge::MergeKeys(
        thrust::raw_pointer_cast(temp_storage.data()),
        temp_storage_bytes,
        keys1.begin(),
        static_cast<int>(keys1.size()),
        keys2.begin(),
        static_cast<int>(keys2.size()),
        result.begin(),
        cuda::std::less<>{},
        stream));
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  };

  SECTION("nullptr")
  {
    merge_keys_on(nullptr);
  }

  SECTION("literal 0")
  {
    merge_keys_on(0);
  }

  const c2h::device_vector<int> expected{0, 0, 2, 3, 3, 4, 5};
  REQUIRE(result == expected);
}

// Runs the two-phase overload wrapped by two_phase with the given environment: queries the temporary storage size
// and executes with user provided storage.
template <class TwoPhaseFn, class EnvT>
void run_two_phase(TwoPhaseFn two_phase, const EnvT& env)
{
  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == two_phase(nullptr, temp_storage_bytes, env));

  c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  REQUIRE(cudaSuccess == two_phase(thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, env));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

CUB_TEST("DeviceMerge::MergeKeys can be tuned with user provided memory", "[merge][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys1                               = c2h::device_vector<int>{0, 2, 5};
  auto keys2                               = c2h::device_vector<int>{0, 3, 3, 4};
  auto result                              = c2h::device_vector<int>(7);
  auto d_block_size                        = c2h::device_vector<unsigned int>(1);

  const block_size_extracting_less_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  run_two_phase(
    [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
      return cub::DeviceMerge::MergeKeys(
        d_temp_storage,
        temp_storage_bytes,
        keys1.begin(),
        static_cast<int>(keys1.size()),
        keys2.begin(),
        static_cast<int>(keys2.size()),
        result.begin(),
        block_size_check,
        env);
    },
    cuda::execution::tune(merge_tuning<target_block_size>{}));

  const c2h::device_vector<int> expected{0, 0, 2, 3, 3, 4, 5};
  REQUIRE(result == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceMerge::MergePairs can be tuned with user provided memory", "[merge][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  auto keys1                               = c2h::device_vector<int>{0, 2, 5};
  auto values1                             = c2h::device_vector<char>{'a', 'b', 'c'};
  auto keys2                               = c2h::device_vector<int>{0, 3, 3, 4};
  auto values2                             = c2h::device_vector<char>{'A', 'B', 'C', 'D'};
  auto result_keys                         = c2h::device_vector<int>(7);
  auto result_values                       = c2h::device_vector<char>(7);
  auto d_block_size                        = c2h::device_vector<unsigned int>(1);

  const block_size_extracting_less_t block_size_check{thrust::raw_pointer_cast(d_block_size.data())};

  run_two_phase(
    [&](void* d_temp_storage, size_t& temp_storage_bytes, const auto& env) {
      return cub::DeviceMerge::MergePairs(
        d_temp_storage,
        temp_storage_bytes,
        keys1.begin(),
        values1.begin(),
        static_cast<int>(keys1.size()),
        keys2.begin(),
        values2.begin(),
        static_cast<int>(keys2.size()),
        result_keys.begin(),
        result_values.begin(),
        block_size_check,
        env);
    },
    cuda::execution::tune(merge_tuning<target_block_size>{}));

  const c2h::device_vector<int> expected_keys{0, 0, 2, 3, 3, 4, 5};
  const c2h::device_vector<char> expected_values{'a', 'A', 'b', 'B', 'C', 'D', 'c'};
  REQUIRE(result_keys == expected_keys);
  REQUIRE(result_values == expected_values);
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH == 0

#if _CCCL_COMPILER(GCC, >=, 8) // gcc 7 cannot preserve constexpr-ness from p1 to p2
CUB_TEST("Test MergePolicy properties", "[merge][device]", CUB_SMALL)
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::MergePolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::MergePolicy>);

  // aggregate init
  constexpr auto p1 = cub::MergePolicy{
    128, 7, cub::CacheLoadModifier::LOAD_LDG, cub::BlockStoreAlgorithm::BLOCK_STORE_WARP_TRANSPOSE, true, false, false};

#  if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto p2 = cub::MergePolicy{
    .threads_per_block        = 128,
    .items_per_thread         = 7,
    .load_modifier            = cub::CacheLoadModifier::LOAD_LDG,
    .store_algorithm          = cub::BlockStoreAlgorithm::BLOCK_STORE_WARP_TRANSPOSE,
    .use_bulk_copy_for_keys   = true,
    .use_bulk_copy_for_values = false,
    .unroll                   = false};
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
          == "MergePolicy { .threads_per_block = 128, .items_per_thread = 7, .load_modifier = LOAD_LDG"
             ", .store_algorithm = BLOCK_STORE_WARP_TRANSPOSE, .use_bulk_copy_for_keys = 1"
             ", .use_bulk_copy_for_values = 0, .unroll = 0 }");
}
#endif // _CCCL_COMPILER(GCC, >=, 8)
