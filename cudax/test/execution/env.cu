//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>

#include <cuda/experimental/__async/queries.cuh>
#include <cuda/experimental/execution.cuh>

#include "cuda/__memory_resource/properties.h"
#include "cuda/experimental/__memory_resource/any_resource.cuh"
#include "cuda/experimental/__memory_resource/device_memory_resource.cuh"
#include "cuda/std/__type_traits/is_constructible.h"
#include <catch2/catch.hpp>

namespace cudax = cuda::experimental;
using env_t     = cudax::execution::env_t<cuda::mr::device_accessible>;

struct test_resource
{
  void* allocate(size_t, size_t)
  {
    return nullptr;
  }
  void* allocate_async(size_t, size_t, cuda::stream_ref)
  {
    return nullptr;
  }
  void deallocate(void*, size_t, size_t) {}
  void deallocate_async(void*, size_t, size_t, cuda::stream_ref) {}

  constexpr bool operator==(const test_resource&) const noexcept
  {
    return true;
  }

  constexpr bool operator!=(const test_resource&) const noexcept
  {
    return false;
  }

  friend void get_property(const test_resource&, cuda::mr::device_accessible) noexcept {}
};

TEST_CASE("env_t is queryable for all properties we want", "[execution, env]")
{
  STATIC_REQUIRE(cudax::__async::__queryable<env_t, cudax::get_stream_t>);
  STATIC_REQUIRE(cudax::__async::__queryable<env_t, cudax::get_memory_resource_t>);
  STATIC_REQUIRE(cudax::__async::__queryable<env_t, cudax::execution::get_execution_policy_t>);
}

TEST_CASE("env_t is default constructible", "[execution, env]")
{
  env_t env;
  CHECK(env.query(cudax::get_stream) == ::cuda::stream_ref{});
  CHECK(env.query(cudax::execution::get_execution_policy) == cudax::execution::execution_policy::unsequenced_device);
  CHECK(env.query(cudax::get_memory_resource) == cudax::mr::device_memory_resource{});
}

TEST_CASE("env_t is constructible from an any_resource", "[execution, env]")
{
  const cudax::mr::any_async_resource<cuda::mr::device_accessible> mr{test_resource{}};

  SECTION("Passing an any_resource")
  {
    env_t env{mr};
    CHECK(env.query(cudax::get_stream) == ::cuda::stream_ref{});
    CHECK(env.query(cudax::execution::get_execution_policy) == cudax::execution::execution_policy::unsequenced_device);
    CHECK(env.query(cudax::get_memory_resource) == mr);
  }

  SECTION("Passing an any_resource and a stream")
  {
    cudax::stream stream{};
    env_t env{mr, stream};
    CHECK(env.query(cudax::get_stream) == stream);
    CHECK(env.query(cudax::execution::get_execution_policy) == cudax::execution::execution_policy::unsequenced_device);
    CHECK(env.query(cudax::get_memory_resource) == mr);
  }

  SECTION("Passing an any_resource, a stream and a policy")
  {
    cudax::stream stream{};
    env_t env{mr, stream, cudax::execution::execution_policy::parallel_unsequenced_device};
    CHECK(env.query(cudax::get_stream) == stream);
    CHECK(env.query(cudax::execution::get_execution_policy)
          == cudax::execution::execution_policy::parallel_unsequenced_device);
    CHECK(env.query(cudax::get_memory_resource) == mr);
  }
}

TEST_CASE("env_t is constructible from an any_resource passed as an rvalue", "[execution, env]")
{
  SECTION("Passing an any_resource")
  {
    env_t env{cudax::mr::any_async_resource<cuda::mr::device_accessible>{test_resource{}}};
    CHECK(env.query(cudax::get_stream) == ::cuda::stream_ref{});
    CHECK(env.query(cudax::execution::get_execution_policy) == cudax::execution::execution_policy::unsequenced_device);
    CHECK(env.query(cudax::get_memory_resource)
          == cudax::mr::any_async_resource<cuda::mr::device_accessible>{test_resource{}});
  }

  SECTION("Passing an any_resource and a stream")
  {
    cudax::stream stream{};
    env_t env{cudax::mr::any_async_resource<cuda::mr::device_accessible>{test_resource{}}, stream};
    CHECK(env.query(cudax::get_stream) == stream);
    CHECK(env.query(cudax::execution::get_execution_policy) == cudax::execution::execution_policy::unsequenced_device);
    CHECK(env.query(cudax::get_memory_resource)
          == cudax::mr::any_async_resource<cuda::mr::device_accessible>{test_resource{}});
  }

  SECTION("Passing an any_resource, a stream and a policy")
  {
    cudax::stream stream{};
    env_t env{cudax::mr::any_async_resource<cuda::mr::device_accessible>{test_resource{}},
              stream,
              cudax::execution::execution_policy::parallel_unsequenced_device};
    CHECK(env.query(cudax::get_stream) == stream);
    CHECK(env.query(cudax::execution::get_execution_policy)
          == cudax::execution::execution_policy::parallel_unsequenced_device);
    CHECK(env.query(cudax::get_memory_resource)
          == cudax::mr::any_async_resource<cuda::mr::device_accessible>{test_resource{}});
  }
}

TEST_CASE("env_t is constructible from a resource", "[execution, env]")
{
  test_resource mr{};

  SECTION("Passing an any_resource")
  {
    env_t env{mr};
    CHECK(env.query(cudax::get_stream) == ::cuda::stream_ref{});
    CHECK(env.query(cudax::execution::get_execution_policy) == cudax::execution::execution_policy::unsequenced_device);
    CHECK(env.query(cudax::get_memory_resource) == mr);
  }

  SECTION("Passing an any_resource and a stream")
  {
    cudax::stream stream{};
    env_t env{mr, stream};
    CHECK(env.query(cudax::get_stream) == stream);
    CHECK(env.query(cudax::execution::get_execution_policy) == cudax::execution::execution_policy::unsequenced_device);
    CHECK(env.query(cudax::get_memory_resource) == mr);
  }

  SECTION("Passing an any_resource, a stream and a policy")
  {
    cudax::stream stream{};
    env_t env{mr, stream, cudax::execution::execution_policy::parallel_unsequenced_device};
    CHECK(env.query(cudax::get_stream) == stream);
    CHECK(env.query(cudax::execution::get_execution_policy)
          == cudax::execution::execution_policy::parallel_unsequenced_device);
    CHECK(env.query(cudax::get_memory_resource) == mr);
  }
}

TEST_CASE("env_t is constructible from a resource passed as an rvalue", "[execution, env]")
{
  SECTION("Passing an any_resource")
  {
    env_t env{test_resource{}};
    CHECK(env.query(cudax::get_stream) == ::cuda::stream_ref{});
    CHECK(env.query(cudax::execution::get_execution_policy) == cudax::execution::execution_policy::unsequenced_device);
    CHECK(env.query(cudax::get_memory_resource) == test_resource{});
  }

  SECTION("Passing an any_resource and a stream")
  {
    cudax::stream stream{};
    env_t env{test_resource{}, stream};
    CHECK(env.query(cudax::get_stream) == stream);
    CHECK(env.query(cudax::execution::get_execution_policy) == cudax::execution::execution_policy::unsequenced_device);
    CHECK(env.query(cudax::get_memory_resource) == test_resource{});
  }

  SECTION("Passing an any_resource, a stream and a policy")
  {
    cudax::stream stream{};
    env_t env{test_resource{}, stream, cudax::execution::execution_policy::parallel_unsequenced_device};
    CHECK(env.query(cudax::get_stream) == stream);
    CHECK(env.query(cudax::execution::get_execution_policy)
          == cudax::execution::execution_policy::parallel_unsequenced_device);
    CHECK(env.query(cudax::get_memory_resource) == test_resource{});
  }
}

struct some_env_t
{
  test_resource res_{};
  cudax::stream stream_{};
  cudax::execution::execution_policy policy_ = cudax::execution::execution_policy::parallel_unsequenced_device;

  const test_resource& query(cudax::get_memory_resource_t) const noexcept
  {
    return res_;
  }

  cudax::stream_ref query(cudax::get_stream_t) const noexcept
  {
    return stream_;
  }

  cudax::execution::execution_policy query(cudax::execution::get_execution_policy_t) const noexcept
  {
    return policy_;
  }
};
TEST_CASE("env_t is constructible from a suitable env", "[execution, env]")
{
  some_env_t other_env{};
  env_t env{other_env};
  CHECK(env.query(cudax::get_stream) == other_env.stream_);
  CHECK(env.query(cudax::execution::get_execution_policy) == other_env.policy_);
  CHECK(env.query(cudax::get_memory_resource) == other_env.res_);
}

template <bool WithResource, bool WithStream, bool WithPolicy>
struct bad_env_t
{
  test_resource res_{};
  cudax::stream stream_{};
  cudax::execution::execution_policy policy_ = cudax::execution::execution_policy::parallel_unsequenced_device;

  template <bool Enable = WithResource, cuda::std::enable_if_t<Enable, int> = 0>
  const test_resource& query(cudax::get_memory_resource_t) const noexcept
  {
    return res_;
  }

  template <bool Enable = WithStream, cuda::std::enable_if_t<Enable, int> = 0>
  cudax::stream_ref query(cudax::get_stream_t) const noexcept
  {
    return stream_;
  }

  template <bool Enable = WithPolicy, cuda::std::enable_if_t<Enable, int> = 0>
  cudax::execution::execution_policy query(cudax::execution::get_execution_policy_t) const noexcept
  {
    return policy_;
  }
};
TEST_CASE("env_t is not constructible from a env missing querries", "[execution, env]")
{
  STATIC_REQUIRE(cuda::std::is_constructible_v<env_t, bad_env_t<true, true, true>>);
  STATIC_REQUIRE(!cuda::std::is_constructible_v<env_t, bad_env_t<false, true, true>>);
  STATIC_REQUIRE(!cuda::std::is_constructible_v<env_t, bad_env_t<true, false, true>>);
  STATIC_REQUIRE(!cuda::std::is_constructible_v<env_t, bad_env_t<true, true, false>>);
}
