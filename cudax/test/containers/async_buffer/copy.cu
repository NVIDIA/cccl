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

// TODO: only device accessible resource
TEMPLATE_TEST_CASE("cudax::async_buffer copy_to",
                   "[container][async_buffer]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Env      = typename extract_properties<TestType>::env;
  using Resource = typename extract_properties<TestType>::resource;
  using Buffer   = typename extract_properties<TestType>::async_buffer;
  using T        = typename Buffer::value_type;

  cudax::stream stream{};
  Resource resource{};
  Env env{resource, stream};

  using MatchingResource = typename extract_properties<TestType>::matching_resource;
  Env matching_env{MatchingResource{resource}, stream};

  SECTION("Same resource and stream")
  {
    { // empty input
      const Buffer input{env};
      const Buffer buf = cudax::copy_to(input, input.get_memory_resource(), input.get_stream());
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const Buffer buf = cudax::copy_to(input, input.get_memory_resource(), input.get_stream());
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }

    { // empty input
      const Buffer input{env};
      const Buffer buf = cudax::copy_to(input);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const Buffer buf = cudax::copy_to(input);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("Different stream")
  {
    cudax::stream other_stream{};
    { // empty input
      const Buffer input{env};
      const Buffer buf = cudax::copy_to(input, input.get_memory_resource(), other_stream);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const Buffer buf = cudax::copy_to(input, input.get_memory_resource(), other_stream);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("Different resource and stream")
  {
    cudax::stream other_stream{};
    { // empty input
      const Buffer input{env};
      const auto buf = cudax::copy_to(input, env.query(cudax::get_memory_resource), other_stream);
      static_assert(!cuda::std::is_same_v<Buffer, cuda::std::remove_const_t<decltype(buf)>>);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const auto buf = cudax::copy_to(input, env.query(cudax::get_memory_resource), other_stream);
      static_assert(!cuda::std::is_same_v<Buffer, cuda::std::remove_const_t<decltype(buf)>>);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("Different resource, same stream")
  {
    { // empty input
      const Buffer input{env};
      const auto buf = cudax::copy_to(input, env.query(cudax::get_memory_resource), stream);
      static_assert(!cuda::std::is_same_v<Buffer, cuda::std::remove_const_t<decltype(buf)>>);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const auto buf = cudax::copy_to(input, env.query(cudax::get_memory_resource), stream);
      static_assert(!cuda::std::is_same_v<Buffer, cuda::std::remove_const_t<decltype(buf)>>);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }

    { // empty input
      const Buffer input{env};
      const auto buf = cudax::copy_to(input, env.query(cudax::get_memory_resource));
      static_assert(!cuda::std::is_same_v<Buffer, cuda::std::remove_const_t<decltype(buf)>>);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const auto buf = cudax::copy_to(input, env.query(cudax::get_memory_resource));
      static_assert(!cuda::std::is_same_v<Buffer, cuda::std::remove_const_t<decltype(buf)>>);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }
}
