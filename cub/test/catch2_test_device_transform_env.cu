// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_transform.cuh>

#include <cuda/iterator>

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

using namespace thrust::placeholders;

auto make_stream_env(cudaStream_t stream)
{
  // MSVC has trouble nesting two aggregate initializations with CTAD
  auto stream_prop = stdexec::prop{cuda::get_stream, cuda::stream_ref{stream}};
  return stdexec::env{cuda::std::move(stream_prop)};
}

C2H_TEST("DeviceTransform::Transform custom stream", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  cuda::constant_iterator<type> a{13};
  cuda::counting_iterator<type> b{42};

  cudaStream_t stream;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  c2h::device_vector<type> result(num_items, thrust::no_init);
  SECTION("raw stream")
  {
    cub::DeviceTransform::Transform(cuda::std::make_tuple(a, b), result.begin(), num_items, _1 + _2, stream);
  }
  SECTION("environment")
  {
    auto env = make_stream_env(stream);
    cub::DeviceTransform::Transform(cuda::std::make_tuple(a, b), result.begin(), num_items, _1 + _2, env);
  }

  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
  REQUIRE(thrust::equal(result.begin(), result.end(), cuda::counting_iterator<type>{42 + 13}));
}

C2H_TEST("DeviceTransform::Transform (single argument) custom stream", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  cuda::counting_iterator<type> a{42};

  cudaStream_t stream;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  c2h::device_vector<type> result(num_items, thrust::no_init);
  SECTION("raw stream")
  {
    cub::DeviceTransform::Transform(a, result.begin(), num_items, _1 + 13, stream);
  }
  SECTION("environment")
  {
    auto env = make_stream_env(stream);
    cub::DeviceTransform::Transform(a, result.begin(), num_items, _1 + 13, env);
  }

  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
  REQUIRE(thrust::equal(result.begin(), result.end(), cuda::counting_iterator<type>{42 + 13}));
}

C2H_TEST("DeviceTransform::Generate custom stream", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  auto generator      = cub::detail::__return_constant<type>{1337};

  cudaStream_t stream;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  c2h::device_vector<type> result(num_items, thrust::no_init);
  SECTION("raw stream")
  {
    cub::DeviceTransform::Generate(result.begin(), num_items, generator, stream);
  }
  SECTION("environment")
  {
    auto env = make_stream_env(stream);
    cub::DeviceTransform::Generate(result.begin(), num_items, generator, env);
  }

  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
  REQUIRE(thrust::equal(result.begin(), result.end(), cuda::constant_iterator<type>{1337}));
}

C2H_TEST("DeviceTransform::Fill custom stream", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths

  cudaStream_t stream;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  c2h::device_vector<type> result(num_items, thrust::no_init);
  SECTION("raw stream")
  {
    cub::DeviceTransform::Fill(result.begin(), num_items, 0xBAD, stream);
  }
  SECTION("environment")
  {
    auto env = make_stream_env(stream);
    cub::DeviceTransform::Fill(result.begin(), num_items, 0xBAD, env);
  }

  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
  REQUIRE(thrust::equal(result.begin(), result.end(), cuda::constant_iterator<type>{0xBAD}));
}

struct reference_func
{
  _CCCL_HOST_DEVICE int operator()(int i) const
  {
    const auto sum = i + 13;
    return sum > 1000 ? sum : 1337;
  }
};

C2H_TEST("DeviceTransform::TransformIf custom stream", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  cuda::constant_iterator<type> a{13};
  cuda::counting_iterator<type> b{42};

  cudaStream_t stream;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  c2h::device_vector<type> result(num_items, 1337);
  SECTION("raw stream")
  {
    cub::DeviceTransform::TransformIf(
      cuda::std::make_tuple(a, b), result.begin(), num_items, (_1 + _2) > 1000, _1 + _2, stream);
  }
  SECTION("environment")
  {
    auto env = make_stream_env(stream);
    cub::DeviceTransform::TransformIf(
      cuda::std::make_tuple(a, b), result.begin(), num_items, (_1 + _2) > 1000, _1 + _2, env);
  }

  auto reference_it = cuda::transform_iterator{cuda::counting_iterator{42}, reference_func{}};

  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
  REQUIRE(thrust::equal(result.begin(), result.end(), reference_it));
}

C2H_TEST("DeviceTransform::TransformIf (single argument) custom stream", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  cuda::counting_iterator<type> a{42};

  cudaStream_t stream;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  c2h::device_vector<type> result(num_items, 1337);
  SECTION("raw stream")
  {
    cub::DeviceTransform::TransformIf(a, result.begin(), num_items, (_1 + 13) > 1000, _1 + 13, stream);
  }
  SECTION("environment")
  {
    auto env = make_stream_env(stream);
    cub::DeviceTransform::TransformIf(a, result.begin(), num_items, (_1 + 13) > 1000, _1 + 13, env);
  }

  auto reference_it = cuda::transform_iterator{cuda::counting_iterator{42}, reference_func{}};

  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
  REQUIRE(thrust::equal(result.begin(), result.end(), reference_it));
}

C2H_TEST("DeviceTransform::TransformStableArgumentAddresses custom stream", "[device][transform]")
{
  using type          = int;
  const int num_items = GENERATE(100, 100'000); // try to hit the small and full tile code paths
  cuda::constant_iterator<type> a{13};
  cuda::counting_iterator<type> b{42};

  cudaStream_t stream;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  c2h::device_vector<type> result(num_items, thrust::no_init);
  SECTION("raw stream")
  {
    cub::DeviceTransform::TransformStableArgumentAddresses(
      cuda::std::make_tuple(a, b), result.begin(), num_items, _1 + _2, stream);
  }
  SECTION("environment")
  {
    auto env = make_stream_env(stream);
    cub::DeviceTransform::TransformStableArgumentAddresses(
      cuda::std::make_tuple(a, b), result.begin(), num_items, _1 + _2, env);
  }

  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
  REQUIRE(thrust::equal(result.begin(), result.end(), cuda::counting_iterator<type>{42 + 13}));
}
