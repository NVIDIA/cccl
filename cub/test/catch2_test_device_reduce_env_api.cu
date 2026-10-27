// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tune.h>
#include <cuda/devices>
#include <cuda/execution>
#include <cuda/memory_resource>
#include <cuda/std/__execution/env.h>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include "catch2_test_memory_resources.h"
#include "cub_test_macros.h"

struct custom_stream_env
{
  cuda::stream_ref stream;

  cuda::stream_ref query(cuda::get_stream_t) const noexcept
  {
    return stream;
  }
};

struct other_query_t
{};

struct custom_resource_env
{
  device_memory_resource resource;

  device_memory_resource query(cuda::mr::get_memory_resource_t) const noexcept
  {
    return resource;
  }
};

using gpu_to_gpu_requirements = cuda::std::execution::env<cuda::execution::determinism::gpu_to_gpu_t>;

struct custom_requirements_env : custom_stream_env
{
  using custom_stream_env::query;

  gpu_to_gpu_requirements query(cuda::execution::__get_requirements_t) const noexcept
  {
    return {};
  }
};

CUB_TEST("DeviceReduce dispatch preserves unadvertised streams", "[reduce][env]", CUB_SMALL)
{
  const cuda::stream_ref expected{cudaStreamPerThread};
  const auto check = [&](const auto& env) {
    cudaStream_t observed{};
    size_t bytes{};
    const auto algorithm = [&](auto, void*, size_t&, cudaStream_t stream) {
      observed = stream;
      return cudaSuccess;
    };
    REQUIRE(cub::detail::dispatch_with_env_validation(nullptr, bytes, env, algorithm) == cudaSuccess);
    REQUIRE(observed == expected.get());
  };

  const custom_stream_env raw{expected};
  const cuda::std::execution::env composed{raw};
  const cuda::std::execution::env mixed{cuda::std::execution::prop{other_query_t{}, 42}, raw};
  const cuda::std::execution::env nested{mixed};
  check(raw);
  check(composed);
  check(mixed);
  check(nested);
  check(cuda::checked_env<other_query_t>(mixed));
  check(cuda::checked_env<>(raw));

  const auto fallback =
    cub::detail::query_or<true>(cuda::get_stream, cuda::stream_ref{cudaStream_t{}}, cuda::std::execution::env<>{});
  REQUIRE(fallback.get() == cudaStream_t{});
}

CUB_TEST("DeviceReduce dispatch preserves unadvertised resources", "[reduce][env]", CUB_SMALL)
{
  size_t bytes_allocated{};
  size_t bytes_deallocated{};
  const device_memory_resource expected{cudaStreamPerThread, &bytes_allocated, &bytes_deallocated};
  const auto check = [&](const auto& env) {
    const auto resource =
      cub::detail::query_or<true>(cuda::mr::get_memory_resource, cub::detail::device_memory_resource{}, env);
    static_assert(cuda::std::is_same_v<cuda::std::remove_cvref_t<decltype(resource)>, device_memory_resource>);
    REQUIRE(resource == expected);
  };

  const custom_resource_env raw{expected};
  const cuda::std::execution::env composed{raw};
  const cuda::std::execution::env mixed{cuda::stream_ref{cudaStreamPerThread}, raw};
  const cuda::std::execution::env nested{mixed};
  check(raw);
  check(composed);
  check(mixed);
  check(nested);
  check(cuda::checked_env<cuda::get_stream_t>(mixed));
  check(cuda::checked_env<>(raw));
}

CUB_TEST("DeviceReduce preserves unadvertised determinism requirements", "[reduce][env]", CUB_SMALL)
{
  const auto check = [](const auto& env) {
    using env_type     = cuda::std::remove_cvref_t<decltype(env)>;
    using requirements = cuda::std::execution::__detail::
      __advertised_query_result_or_t<env_type, cuda::execution::__get_requirements_t, cuda::std::execution::env<>>;
    using determinism =
      cuda::std::execution::__query_result_or_t<requirements,
                                                cuda::execution::determinism::__get_determinism_t,
                                                cuda::execution::determinism::run_to_run_t>;
    static_assert(cuda::std::is_same_v<determinism, cuda::execution::determinism::gpu_to_gpu_t>);
  };

  const custom_requirements_env raw{{cuda::stream_ref{cudaStreamPerThread}}};
  const cuda::std::execution::env composed{raw};
  const cuda::std::execution::env mixed{cuda::stream_ref{cudaStreamPerThread}, raw};
  const cuda::std::execution::env nested{mixed};
  check(raw);
  check(composed);
  check(mixed);
  check(nested);
  check(cuda::checked_env<cuda::get_stream_t>(mixed));
  check(cuda::checked_env<>(raw));
}

CUB_TEST("cub::DeviceReduce::Reduce accepts run_to_run determinism requirements", "[reduce][env]", CUB_SMALL)
{
  // TODO(srinivas): replace with gpu_to_gpu once offset size restriction is relaxed
  // example-begin reduce-env-determinism
  auto op     = cuda::std::plus{};
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);
  auto init   = 0.0f;

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error = cub::DeviceReduce::Reduce(input.begin(), output.begin(), input.size(), op, init, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Reduce failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected{6.0f};
  // example-end reduce-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

CUB_TEST("cub::DeviceReduce::Reduce accepts not_guaranteed determinism requirements", "[reduce][env]", CUB_SMALL)
{
  // example-begin reduce-env-non-determinism
  auto op     = cuda::std::plus{};
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);
  auto init   = 0.0f;

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error = cub::DeviceReduce::Reduce(input.begin(), output.begin(), input.size(), op, init, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Reduce failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected{6.0f};
  // example-end reduce-env-non-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

CUB_TEST("cub::DeviceReduce::Reduce accepts stream", "[reduce][env]", CUB_SMALL)
{
  // example-begin reduce-env-stream
  auto op     = cuda::std::plus{};
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);
  auto init   = 0.0f;

  const cuda::stream stream{cuda::devices[0]};
  const cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceReduce::Reduce(input.begin(), output.begin(), input.size(), op, init, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Reduce failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected{6.0f};
  // example-end reduce-env-stream
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

CUB_TEST("cub::DeviceReduce::Reduce accepts a checked custom environment", "[reduce][env]", CUB_SMALL)
{
  auto input  = thrust::device_vector<int>{1, 2, 3, 4};
  auto output = thrust::device_vector<int>(1);

  const cuda::stream stream{cuda::devices[0]};
  auto env = cuda::checked_env<cuda::get_stream_t>(custom_stream_env{cuda::stream_ref{stream}});

  const auto error =
    cub::DeviceReduce::Reduce(input.begin(), output.begin(), input.size(), cuda::std::plus<>{}, 0, env);

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(output[0] == 10);
}

CUB_TEST("cub::DeviceReduce::Reduce accepts a stream object", "[reduce][env]", CUB_SMALL)
{
  auto input  = thrust::device_vector<int>{1, 2, 3, 4};
  auto output = thrust::device_vector<int>(1);

  const cuda::stream stream{cuda::devices[0]};
  const auto error =
    cub::DeviceReduce::Reduce(input.begin(), output.begin(), input.size(), cuda::std::plus<>{}, 0, stream.get());

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(output[0] == 10);
}

CUB_TEST("cub::DeviceReduce::Sum accepts run_to_run determinism requirements", "[reduce][env]", CUB_SMALL)
{
  // TODO(srinivas): replace with gpu_to_gpu once offset size restriction is relaxed
  // example-begin sum-env-determinism
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error = cub::DeviceReduce::Sum(input.begin(), output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Sum failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected{6.0f};
  // example-end sum-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

CUB_TEST("cub::DeviceReduce::Sum accepts not_guaranteed determinism requirements", "[reduce][env]", CUB_SMALL)
{
  // example-begin sum-env-non-determinism
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error = cub::DeviceReduce::Sum(input.begin(), output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Sum failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected{6.0f};
  // example-end sum-env-non-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

CUB_TEST("cub::DeviceReduce::Sum accepts stream", "[reduce][env]", CUB_SMALL)
{
  // example-begin sum-env-stream
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  const cuda::stream stream{cuda::devices[0]};
  const cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceReduce::Sum(input.begin(), output.begin(), input.size(), stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Sum failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected{6.0f};
  // example-end sum-env-stream
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

CUB_TEST("cub::DeviceReduce::Min accepts run_to_run determinism requirements", "[reduce][env]", CUB_SMALL)
{
  // example-begin min-env-determinism
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error = cub::DeviceReduce::Min(input.begin(), output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Min failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected{0.0f};
  // example-end min-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

CUB_TEST("cub::DeviceReduce::Min accepts not_guaranteed determinism requirements", "[reduce][env]", CUB_SMALL)
{
  // example-begin min-env-non-determinism
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error = cub::DeviceReduce::Min(input.begin(), output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Min failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected{0.0f};
  // example-end min-env-non-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

CUB_TEST("cub::DeviceReduce::Min accepts stream", "[reduce][env]", CUB_SMALL)
{
  // example-begin min-env-stream
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  const cuda::stream stream{cuda::devices[0]};
  const cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceReduce::Min(input.begin(), output.begin(), input.size(), stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Min failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected{0.0f};
  // example-end min-env-stream
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

CUB_TEST("cub::DeviceReduce::Max accepts run_to_run determinism requirements", "[reduce][env]", CUB_SMALL)
{
  // example-begin max-env-determinism
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error = cub::DeviceReduce::Max(input.begin(), output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Max failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected{3.0f};
  // example-end max-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

CUB_TEST("cub::DeviceReduce::Max accepts not_guaranteed determinism requirements", "[reduce][env]", CUB_SMALL)
{
  // example-begin max-env-non-determinism
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error = cub::DeviceReduce::Max(input.begin(), output.begin(), input.size(), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Max failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected{3.0f};
  // example-end max-env-non-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

CUB_TEST("cub::DeviceReduce::Max accepts stream", "[reduce][env]", CUB_SMALL)
{
  // example-begin max-env-stream
  auto input  = thrust::device_vector<float>{0.0f, 1.0f, 2.0f, 3.0f};
  auto output = thrust::device_vector<float>(1);

  const cuda::stream stream{cuda::devices[0]};
  const cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceReduce::Max(input.begin(), output.begin(), input.size(), stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Max failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected{3.0f};
  // example-end max-env-stream
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

CUB_TEST("cub::DeviceReduce::ArgMin accepts run_to_run determinism requirements", "[reduce][env]", CUB_SMALL)
{
  // example-begin argmin-env-determinism
  auto input        = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_output   = thrust::device_vector<float>(1);
  auto index_output = thrust::device_vector<cuda::std::int64_t>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error = cub::DeviceReduce::ArgMin(
    input.begin(), min_output.begin(), index_output.begin(), static_cast<::cuda::std::int64_t>(input.size()), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ArgMin failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected_min{0.0f};
  const thrust::device_vector<cuda::std::int64_t> expected_index{3};
  // example-end argmin-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(min_output == expected_min);
  REQUIRE(index_output == expected_index);
}

CUB_TEST("cub::DeviceReduce::ArgMin accepts not_guaranteed determinism requirements", "[reduce][env]", CUB_SMALL)
{
  // example-begin argmin-env-non-determinism
  auto input        = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_output   = thrust::device_vector<float>(1);
  auto index_output = thrust::device_vector<cuda::std::int64_t>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error = cub::DeviceReduce::ArgMin(
    input.begin(), min_output.begin(), index_output.begin(), static_cast<::cuda::std::int64_t>(input.size()), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ArgMin failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected_min{0.0f};
  const thrust::device_vector<cuda::std::int64_t> expected_index{3};
  // example-end argmin-env-non-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(min_output == expected_min);
  REQUIRE(index_output == expected_index);
}

CUB_TEST("cub::DeviceReduce::ArgMin accepts stream", "[reduce][env]", CUB_SMALL)
{
  // example-begin argmin-env-stream
  auto input        = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto min_output   = thrust::device_vector<float>(1);
  auto index_output = thrust::device_vector<cuda::std::int64_t>(1);

  const cuda::stream stream{cuda::devices[0]};
  const cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceReduce::ArgMin(
    input.begin(),
    min_output.begin(),
    index_output.begin(),
    static_cast<::cuda::std::int64_t>(input.size()),
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ArgMin failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected_min{0.0f};
  const thrust::device_vector<cuda::std::int64_t> expected_index{3};
  // example-end argmin-env-stream
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(min_output == expected_min);
  REQUIRE(index_output == expected_index);
}

CUB_TEST("cub::DeviceReduce::ArgMax accepts run_to_run determinism requirements", "[reduce][env]", CUB_SMALL)
{
  // example-begin argmax-env-determinism
  auto input        = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto max_output   = thrust::device_vector<float>(1);
  auto index_output = thrust::device_vector<cuda::std::int64_t>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error = cub::DeviceReduce::ArgMax(
    input.begin(), max_output.begin(), index_output.begin(), static_cast<::cuda::std::int64_t>(input.size()), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ArgMax failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected_max{4.0f};
  const thrust::device_vector<cuda::std::int64_t> expected_index{2};
  // example-end argmax-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(max_output == expected_max);
  REQUIRE(index_output == expected_index);
}

CUB_TEST("cub::DeviceReduce::ArgMax accepts not_guaranteed determinism requirements", "[reduce][env]", CUB_SMALL)
{
  // example-begin argmax-env-non-determinism
  auto input        = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto max_output   = thrust::device_vector<float>(1);
  auto index_output = thrust::device_vector<cuda::std::int64_t>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error = cub::DeviceReduce::ArgMax(
    input.begin(), max_output.begin(), index_output.begin(), static_cast<::cuda::std::int64_t>(input.size()), env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ArgMax failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected_max{4.0f};
  const thrust::device_vector<cuda::std::int64_t> expected_index{2};
  // example-end argmax-env-non-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(max_output == expected_max);
  REQUIRE(index_output == expected_index);
}

CUB_TEST("cub::DeviceReduce::ArgMax accepts stream", "[reduce][env]", CUB_SMALL)
{
  // example-begin argmax-env-stream
  auto input        = thrust::device_vector<float>{3.0f, 1.0f, 4.0f, 0.0f, 2.0f};
  auto max_output   = thrust::device_vector<float>(1);
  auto index_output = thrust::device_vector<cuda::std::int64_t>(1);

  const cuda::stream stream{cuda::devices[0]};
  const cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceReduce::ArgMax(
    input.begin(),
    max_output.begin(),
    index_output.begin(),
    static_cast<::cuda::std::int64_t>(input.size()),
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ArgMax failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected_max{4.0f};
  const thrust::device_vector<cuda::std::int64_t> expected_index{2};
  // example-end argmax-env-stream

  REQUIRE(error == cudaSuccess);
  REQUIRE(max_output == expected_max);
  REQUIRE(index_output == expected_index);
}

CUB_TEST("cub::DeviceReduce::TransformReduce accepts determinism requirements", "[reduce][env]", CUB_SMALL)
{
  // example-begin transform-reduce-env-determinism
  auto op        = cuda::std::plus{};
  auto transform = cuda::std::negate{};
  auto input     = thrust::device_vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
  auto output    = thrust::device_vector<float>(1);
  auto init      = 0.0f;

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error =
    cub::DeviceReduce::TransformReduce(input.begin(), output.begin(), input.size(), op, transform, init, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::TransformReduce failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected{-10.0f};
  // example-end transform-reduce-env-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

CUB_TEST("cub::DeviceReduce::TransformReduce accepts not_guaranteed determinism requirements",
         "[reduce][env]",
         CUB_SMALL)
{
  // example-begin transform-reduce-env-non-determinism
  auto op        = cuda::std::plus{};
  auto transform = cuda::std::negate{};
  auto input     = thrust::device_vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
  auto output    = thrust::device_vector<float>(1);
  auto init      = 0.0f;

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error =
    cub::DeviceReduce::TransformReduce(input.begin(), output.begin(), input.size(), op, transform, init, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::TransformReduce failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected{-10.0f};
  // example-end transform-reduce-env-non-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

CUB_TEST("cub::DeviceReduce::TransformReduce accepts stream", "[reduce][env]", CUB_SMALL)
{
  // example-begin transform-reduce-env-stream
  auto op        = cuda::std::plus{};
  auto transform = cuda::std::negate{};
  auto input     = thrust::device_vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
  auto output    = thrust::device_vector<float>(1);
  auto init      = 0.0f;

  const cuda::stream stream{cuda::devices[0]};
  const cuda::stream_ref stream_ref{stream};

  auto error =
    cub::DeviceReduce::TransformReduce(input.begin(), output.begin(), input.size(), op, transform, init, stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::TransformReduce failed with status: " << error << '\n';
  }

  const thrust::device_vector<float> expected{-10.0f};
  // example-end transform-reduce-env-stream
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

CUB_TEST("cub::DeviceReduce::ReduceByKey accepts run_to_run determinism requirements", "[reduce][env]", CUB_SMALL)
{
  // example-begin reduce-by-key-env
  auto keys_in         = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto values_in       = thrust::device_vector<int>{0, 7, 1, 6, 2, 5, 3, 4};
  auto unique_keys_out = thrust::device_vector<int>(5);
  auto aggregates_out  = thrust::device_vector<int>(5);
  auto num_runs_out    = thrust::device_vector<int>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

  auto error = cub::DeviceReduce::ReduceByKey(
    keys_in.begin(),
    unique_keys_out.begin(),
    values_in.begin(),
    aggregates_out.begin(),
    num_runs_out.begin(),
    cuda::minimum<int>{},
    static_cast<int>(keys_in.size()),
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ReduceByKey failed with status: " << error << '\n';
  }

  const thrust::device_vector<int> expected_keys{0, 2, 9, 5, 8};
  const thrust::device_vector<int> expected_aggregates{0, 1, 6, 2, 4};
  // example-end reduce-by-key-env

  REQUIRE(error == cudaSuccess);
  REQUIRE(num_runs_out[0] == 5);
  unique_keys_out.resize(5);
  aggregates_out.resize(5);
  REQUIRE(unique_keys_out == expected_keys);
  REQUIRE(aggregates_out == expected_aggregates);
}

CUB_TEST("cub::DeviceReduce::ReduceByKey accepts not_guaranteed determinism requirements", "[reduce][env]", CUB_SMALL)
{
  // example-begin reduce-by-key-env-non-determinism
  auto keys_in         = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto values_in       = thrust::device_vector<int>{0, 7, 1, 6, 2, 5, 3, 4};
  auto unique_keys_out = thrust::device_vector<int>(5);
  auto aggregates_out  = thrust::device_vector<int>(5);
  auto num_runs_out    = thrust::device_vector<int>(1);

  auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed);

  auto error = cub::DeviceReduce::ReduceByKey(
    keys_in.begin(),
    unique_keys_out.begin(),
    values_in.begin(),
    aggregates_out.begin(),
    num_runs_out.begin(),
    cuda::minimum<int>{},
    static_cast<int>(keys_in.size()),
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ReduceByKey failed with status: " << error << '\n';
  }

  const thrust::device_vector<int> expected_keys{0, 2, 9, 5, 8};
  const thrust::device_vector<int> expected_aggregates{0, 1, 6, 2, 4};
  // example-end reduce-by-key-env-non-determinism

  REQUIRE(error == cudaSuccess);
  REQUIRE(num_runs_out[0] == 5);
  unique_keys_out.resize(5);
  aggregates_out.resize(5);
  REQUIRE(unique_keys_out == expected_keys);
  REQUIRE(aggregates_out == expected_aggregates);
}

CUB_TEST("cub::DeviceReduce::ReduceByKey accepts stream", "[reduce][env]", CUB_SMALL)
{
  // example-begin reduce-by-key-env-stream
  auto keys_in         = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto values_in       = thrust::device_vector<int>{0, 7, 1, 6, 2, 5, 3, 4};
  auto unique_keys_out = thrust::device_vector<int>(5);
  auto aggregates_out  = thrust::device_vector<int>(5);
  auto num_runs_out    = thrust::device_vector<int>(1);

  const cuda::stream stream{cuda::devices[0]};
  const cuda::stream_ref stream_ref{stream};

  auto error = cub::DeviceReduce::ReduceByKey(
    keys_in.begin(),
    unique_keys_out.begin(),
    values_in.begin(),
    aggregates_out.begin(),
    num_runs_out.begin(),
    cuda::minimum<int>{},
    static_cast<int>(keys_in.size()),
    stream_ref);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ReduceByKey failed with status: " << error << '\n';
  }

  const thrust::device_vector<int> expected_keys{0, 2, 9, 5, 8};
  const thrust::device_vector<int> expected_aggregates{0, 1, 6, 2, 4};
  // example-end reduce-by-key-env-stream
  stream.sync();

  REQUIRE(error == cudaSuccess);
  REQUIRE(num_runs_out[0] == 5);
  unique_keys_out.resize(5);
  aggregates_out.resize(5);
  REQUIRE(unique_keys_out == expected_keys);
  REQUIRE(aggregates_out == expected_aggregates);
}

CUB_TEST("cub::DeviceReduce::Sum queries both stream and resource from composed env", "[reduce][env]", CUB_SMALL)
{
  auto input  = thrust::device_vector<int>{1, 2, 3, 4, 5};
  auto output = thrust::device_vector<int>(1);

  const cuda::stream stream{cuda::devices[0]};

  size_t bytes_allocated{};
  size_t bytes_deallocated{};
  auto mr = device_memory_resource{stream.get(), &bytes_allocated, &bytes_deallocated};

  auto env = cuda::std::execution::env{cuda::stream_ref{stream}, mr};

  auto error = cub::DeviceReduce::Sum(input.begin(), output.begin(), static_cast<int>(input.size()), env);

  stream.sync();
  REQUIRE(error == cudaSuccess);
  REQUIRE(output[0] == 15);
  REQUIRE(bytes_allocated > 0);
  REQUIRE(bytes_deallocated == bytes_allocated);
}

CUB_TEST("cub::DeviceReduce::Sum accepts a memory resource", "[reduce][env]", CUB_SMALL)
{
  auto input  = thrust::device_vector<int>{1, 2, 3, 4, 5};
  auto output = thrust::device_vector<int>(1);

  size_t bytes_allocated{};
  size_t bytes_deallocated{};
  auto mr = device_memory_resource{nullptr, &bytes_allocated, &bytes_deallocated};

  const auto error = cub::DeviceReduce::Sum(input.begin(), output.begin(), static_cast<int>(input.size()), mr);

  REQUIRE(error == cudaSuccess);
  REQUIRE(output[0] == 15);
  REQUIRE(bytes_allocated > 0);
  REQUIRE(bytes_deallocated == bytes_allocated);
}

CUB_TEST("cub::DeviceReduce::Sum accepts a memory resource reference", "[reduce][env]", CUB_SMALL)
{
  auto input  = thrust::device_vector<int>{1, 2, 3, 4, 5};
  auto output = thrust::device_vector<int>(1);

  size_t bytes_allocated{};
  size_t bytes_deallocated{};
  auto mr     = device_memory_resource{nullptr, &bytes_allocated, &bytes_deallocated};
  auto mr_ref = cuda::mr::resource_ref<>{mr};

  const auto error = cub::DeviceReduce::Sum(input.begin(), output.begin(), static_cast<int>(input.size()), mr_ref);

  REQUIRE(error == cudaSuccess);
  REQUIRE(output[0] == 15);
  REQUIRE(bytes_allocated > 0);
  REQUIRE(bytes_deallocated == bytes_allocated);
}

#if _CCCL_STD_VER >= 2020

// example-begin reduce-policy-selector
struct ReducePolicySelector
{
  __host__ __device__ constexpr auto operator()(cuda::compute_capability cc) const -> cub::ReducePolicy
  {
    auto pass = cub::ReducePassPolicy{
      .threads_per_block = 256,
      .items_per_thread  = cc > cuda::compute_capability{9, 0} ? 20 : 16,
      .vec_size          = 4,
      .reduce_algorithm  = cub::BLOCK_REDUCE_WARP_REDUCTIONS,
      .load_modifier     = cub::LOAD_LDG};
    return {.multi_tile = pass, .single_tile = pass};
  }
};
// example-end reduce-policy-selector

CUB_TEST("cub::DeviceReduce::Reduce accepts a custom policy selector", "[reduce][env]", CUB_SMALL)
{
  // example-begin reduce-tuning
  auto input  = thrust::device_vector<int>{1, 2, 3, 4, 5};
  auto output = thrust::device_vector<int>(1, thrust::no_init);

  const auto error = cub::DeviceReduce::Reduce(
    input.begin(), output.begin(), input.size(), cuda::std::plus{}, 0, cuda::execution::tune(ReducePolicySelector{}));
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Reduce failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{15};
  // example-end reduce-tuning

  REQUIRE(error == cudaSuccess);
  REQUIRE(output == expected);
}

#endif // _CCCL_STD_VER >= 2020
