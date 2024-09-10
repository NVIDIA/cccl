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
  static_assert(!::cuda::std::is_invocable_v<decltype(::cuda::experimental::get_stream), const non_const_get_stream&>);
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
  static_assert(
    !::cuda::std::is_invocable_v<decltype(::cuda::experimental::get_stream), const get_stream_wrong_return&>);
}
