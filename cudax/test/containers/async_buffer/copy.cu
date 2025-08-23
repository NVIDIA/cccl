//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory_resource>
#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/initializer_list>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/container.cuh>

#include "helper.h"
#include "types.h"

#if _CCCL_CUDACC_AT_LEAST(12, 6)
using test_types = c2h::type_list<cuda::std::tuple<cuda::mr::host_accessible>,
                                  cuda::std::tuple<cuda::mr::device_accessible>,
                                  cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>>;
#else
using test_types = c2h::type_list<cuda::std::tuple<cuda::mr::device_accessible>>;
#endif

C2H_CCCLRT_TEST("cudax::async_buffer make_async_buffer", "[container][async_buffer]", test_types)
{
  using TestT    = c2h::get<0, TestType>;
  using Env      = typename extract_properties<TestT>::env;
  using Resource = typename extract_properties<TestT>::resource;
  using Buffer   = typename extract_properties<TestT>::async_buffer;
  using T        = typename Buffer::value_type;

  cudax::stream stream{cuda::device_ref{0}};
  Resource resource{};
  Env env{resource, stream};

  using MatchingResource = typename extract_properties<TestT>::matching_resource;
  Env matching_env{MatchingResource{resource}, stream};

  SECTION("Same resource and stream")
  {
    { // empty input
      const Buffer input{env};
      const Buffer buf = cudax::make_async_buffer(input.stream(), input.memory_resource(), input);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const Buffer buf = cudax::make_async_buffer(input.stream(), input.memory_resource(), input);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }

    { // empty input
      const Buffer input{env};
      const Buffer buf = cudax::make_async_buffer(input.stream(), input.memory_resource(), input);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const Buffer buf = cudax::make_async_buffer(input.stream(), input.memory_resource(), input);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("Different stream")
  {
    cudax::stream other_stream{cuda::device_ref{0}};
    { // empty input
      const Buffer input{env};
      const Buffer buf = cudax::make_async_buffer(other_stream, input.memory_resource(), input);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const Buffer buf = cudax::make_async_buffer(other_stream, input.memory_resource(), input);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("Different resource and stream")
  {
    cudax::stream other_stream{cuda::device_ref{0}};
    { // empty input
      const Buffer input{env};
      const auto buf = cudax::make_async_buffer(other_stream, env.query(cuda::mr::get_memory_resource), input);
      static_assert(!cuda::std::is_same_v<Buffer, cuda::std::remove_const_t<decltype(buf)>>);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const auto buf = cudax::make_async_buffer(other_stream, env.query(cuda::mr::get_memory_resource), input);
      static_assert(!cuda::std::is_same_v<Buffer, cuda::std::remove_const_t<decltype(buf)>>);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("Different resource, same stream")
  {
    { // empty input
      const Buffer input{env};
      const auto buf = cudax::make_async_buffer(stream, env.query(cuda::mr::get_memory_resource), input);
      static_assert(!cuda::std::is_same_v<Buffer, cuda::std::remove_const_t<decltype(buf)>>);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const auto buf = cudax::make_async_buffer(stream, env.query(cuda::mr::get_memory_resource), input);
      static_assert(!cuda::std::is_same_v<Buffer, cuda::std::remove_const_t<decltype(buf)>>);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }

    { // empty input
      const Buffer input{env};
      const auto buf = cudax::make_async_buffer(stream, env.query(cuda::mr::get_memory_resource), input);
      static_assert(!cuda::std::is_same_v<Buffer, cuda::std::remove_const_t<decltype(buf)>>);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const auto buf = cudax::make_async_buffer(stream, env.query(cuda::mr::get_memory_resource), input);
      static_assert(!cuda::std::is_same_v<Buffer, cuda::std::remove_const_t<decltype(buf)>>);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }
}

C2H_CCCLRT_TEST("make_async_buffer variants", "[container][async_buffer]")
{
  cudax::stream stream{cuda::device_ref{0}};
  cudax::env_t<cuda::mr::device_accessible, other_property> env{
    cudax::device_memory_resource{cuda::device_ref{0}}, stream};
  const cudax::async_buffer<int, cuda::mr::device_accessible, other_property> input{
    env, {int(1), int(42), int(1337), int(0), int(12), int(-1)}};

  // straight from a resource
  auto buf =
    cuda::experimental::make_async_buffer(input.stream(), cudax::device_memory_resource{cuda::device_ref{0}}, input);
  CUDAX_CHECK(equal_range(buf));
  static_assert(
    ::cuda::mr::synchronous_resource_with<typename decltype(buf)::__resource_t, cuda::mr::device_accessible>);
  static_assert(
    !::cuda::mr::synchronous_resource_with<typename decltype(buf)::__resource_t, cuda::mr::host_accessible>);
  static_assert(!::cuda::mr::synchronous_resource_with<typename decltype(buf)::__resource_t, other_property>);

  auto buf2 = cuda::experimental::make_async_buffer<int, cuda::mr::device_accessible>(
    input.stream(), {cudax::device_memory_resource{cuda::device_ref{0}}}, input);
  CUDAX_CHECK(equal_range(buf2));
  static_assert(
    ::cuda::mr::synchronous_resource_with<typename decltype(buf2)::__resource_t, cuda::mr::device_accessible>);
  static_assert(!::cuda::mr::synchronous_resource_with<typename decltype(buf2)::__resource_t, other_property>);
  static_assert(
    !::cuda::mr::synchronous_resource_with<typename decltype(buf2)::__resource_t, cuda::mr::host_accessible>);

  // from any resource
  auto any_res = cudax::any_resource<cuda::mr::device_accessible, other_property>(
    cudax::device_memory_resource{cuda::device_ref{0}});
  auto buf3 = cudax::make_async_buffer(input.stream(), any_res, input);
  CUDAX_CHECK(equal_range(buf3));
  static_assert(
    ::cuda::mr::synchronous_resource_with<typename decltype(buf3)::__resource_t, cuda::mr::device_accessible>);
  static_assert(::cuda::mr::synchronous_resource_with<typename decltype(buf3)::__resource_t, other_property>);
  static_assert(
    !::cuda::mr::synchronous_resource_with<typename decltype(buf3)::__resource_t, cuda::mr::host_accessible>);

  auto buf4 = cudax::make_async_buffer<int, cuda::mr::device_accessible>(input.stream(), {any_res}, input);
  CUDAX_CHECK(equal_range(buf4));
  static_assert(
    ::cuda::mr::synchronous_resource_with<typename decltype(buf4)::__resource_t, cuda::mr::device_accessible>);
  static_assert(!::cuda::mr::synchronous_resource_with<typename decltype(buf4)::__resource_t, other_property>);
  static_assert(
    !::cuda::mr::synchronous_resource_with<typename decltype(buf4)::__resource_t, cuda::mr::host_accessible>);

  // from a resource reference
  auto res_ref = cudax::resource_ref<cuda::mr::device_accessible, other_property>{any_res};
  auto buf5    = cudax::make_async_buffer(input.stream(), res_ref, input);
  CUDAX_CHECK(equal_range(buf5));
  static_assert(
    ::cuda::mr::synchronous_resource_with<typename decltype(buf5)::__resource_t, cuda::mr::device_accessible>);
  static_assert(::cuda::mr::synchronous_resource_with<typename decltype(buf5)::__resource_t, other_property>);
  static_assert(
    !::cuda::mr::synchronous_resource_with<typename decltype(buf5)::__resource_t, cuda::mr::host_accessible>);

  auto buf6 = cudax::make_async_buffer<int, cuda::mr::device_accessible>(input.stream(), {res_ref}, input);
  CUDAX_CHECK(equal_range(buf6));
  static_assert(
    ::cuda::mr::synchronous_resource_with<typename decltype(buf6)::__resource_t, cuda::mr::device_accessible>);
  static_assert(!::cuda::mr::synchronous_resource_with<typename decltype(buf6)::__resource_t, other_property>);
  static_assert(
    !::cuda::mr::synchronous_resource_with<typename decltype(buf6)::__resource_t, cuda::mr::host_accessible>);

  auto shared_res = cudax::make_shared_resource<cudax::device_memory_resource>(cuda::device_ref{0});
  auto buf7       = cudax::make_async_buffer(input.stream(), shared_res, input);
  CUDAX_CHECK(equal_range(buf7));
  static_assert(
    ::cuda::mr::synchronous_resource_with<typename decltype(buf7)::__resource_t, cuda::mr::device_accessible>);
  static_assert(!::cuda::mr::synchronous_resource_with<typename decltype(buf7)::__resource_t, other_property>);
  static_assert(
    !::cuda::mr::synchronous_resource_with<typename decltype(buf7)::__resource_t, cuda::mr::host_accessible>);

  auto buf8 = cudax::make_async_buffer<int, cuda::mr::device_accessible>(input.stream(), {shared_res}, input);
  CUDAX_CHECK(equal_range(buf8));
  static_assert(
    ::cuda::mr::synchronous_resource_with<typename decltype(buf8)::__resource_t, cuda::mr::device_accessible>);
  static_assert(!::cuda::mr::synchronous_resource_with<typename decltype(buf8)::__resource_t, other_property>);
  static_assert(
    !::cuda::mr::synchronous_resource_with<typename decltype(buf8)::__resource_t, cuda::mr::host_accessible>);
}
