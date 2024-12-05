//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/stream.cuh>

#include <catch2/catch.hpp>
#include <utility.cuh>

TEST_CASE("Can call get_stream on a cudaStream_t", "[stream]")
{
  ::cudaStream_t str = nullptr;
  auto ref           = ::cuda::experimental::get_stream(str);
  CUDAX_CHECK(str == ref);
}

TEST_CASE("Can call get_stream on a cudax::stream", "[stream]")
{
  cudax::stream str;
  auto ref = ::cuda::experimental::get_stream(str);
  CUDAX_CHECK(str == ref);
}

struct something_stream_ordered
{
  cudax::stream stream_{};

  ::cuda::stream_ref get_stream() const noexcept
  {
    return stream_;
  }
};

TEST_CASE("Can call get_stream on a type with a get_stream method", "[stream]")
{
  something_stream_ordered str{};
  auto ref = ::cuda::experimental::get_stream(str);
  CUDAX_CHECK(str.stream_ == ref);
}

struct non_const_get_stream
{
  cudax::stream stream_{};

  ::cuda::stream_ref get_stream() noexcept
  {
    return stream_;
  }
};

TEST_CASE("The get_stream method must be const qualified", "[stream]")
{
  STATIC_REQUIRE(!::cuda::std::is_invocable_v<::cuda::experimental::get_stream_t, const non_const_get_stream&>);
}

struct get_stream_wrong_return
{
  ::cudaStream_t stream_{};

  ::cudaStream_t get_stream() const noexcept
  {
    return stream_;
  }
};
TEST_CASE("The get_stream method must return a cuda::stream_ref", "[stream]")
{
  STATIC_REQUIRE(!::cuda::std::is_invocable_v<::cuda::experimental::get_stream_t, const get_stream_wrong_return&>);
}

struct env_with_query
{
  cudax::stream stream_{};

  ::cuda::stream_ref query(::cuda::experimental::get_stream_t) const noexcept
  {
    return ::cuda::stream_ref{stream_.get()};
  }
};
TEST_CASE("Works with queries", "[stream]")
{
  env_with_query env;
  ::cuda::stream_ref ref = ::cuda::experimental::get_stream(env);
  CUDAX_CHECK(ref == env.stream_);
}

struct env_with_query_that_returns_cudastream
{
  cudax::stream stream_{};

  ::cudaStream_t query(::cuda::experimental::get_stream_t) const noexcept
  {
    return stream_.get();
  }
};
TEST_CASE("Works with queries that return cudastream", "[stream]")
{
  env_with_query_that_returns_cudastream env;
  ::cuda::stream_ref ref = ::cuda::experimental::get_stream(env);
  CUDAX_CHECK(ref == env.stream_);
}

struct env_with_query_that_returns_stream
{
  cudax::stream stream_{};

  const cudax::stream& query(::cuda::experimental::get_stream_t) const noexcept
  {
    return stream_;
  }
};
TEST_CASE("Works with queries that return stream", "[stream]")
{
  env_with_query_that_returns_stream env;
  ::cuda::stream_ref ref = ::cuda::experimental::get_stream(env);
  CUDAX_CHECK(ref == env.stream_);
}

struct env_with_query_that_returns_wrong_type
{
  float query(::cuda::experimental::get_stream_t) const noexcept
  {
    return 42;
  }
};
TEST_CASE("Queries require a proper return type", "[stream]")
{
  STATIC_REQUIRE(
    !::cuda::std::is_invocable_v<::cuda::experimental::get_stream_t, const env_with_query_that_returns_wrong_type&>);
}
