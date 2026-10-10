// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/std/execution>
#include <cuda/stream>

#include <c2h/catch2_test_helper.h>
#include <c2h/device_and_stream.h>

struct stream_convertible
{
  cudaStream_t stream;

  operator cudaStream_t() const noexcept
  {
    return stream;
  }
};

struct stream_convertible_non_copyable
{
  cudaStream_t stream;

  stream_convertible_non_copyable(cudaStream_t stream)
      : stream(stream)
  {}

  stream_convertible_non_copyable(const stream_convertible_non_copyable&)                    = delete;
  auto operator=(const stream_convertible_non_copyable&) -> stream_convertible_non_copyable& = delete;
  stream_convertible_non_copyable(stream_convertible_non_copyable&&)                         = default;
  auto operator=(stream_convertible_non_copyable&&) -> stream_convertible_non_copyable&      = default;

  operator cudaStream_t() const noexcept
  {
    return stream;
  }
};

struct with_stream_method
{
  cudaStream_t str;

  auto stream() const noexcept
  {
    return str;
  }
};

struct with_get_stream_method
{
  cudaStream_t stream;

  auto get_stream() const noexcept
  {
    return stream;
  }
};

//! Calls test_func with different versions of the passed stream
template <typename TestFunc>
void test_with_custom_streams(TestFunc&& test_func, const cuda::stream& stream)
{
  SECTION("works with cudaStream_t")
  {
    test_func(stream.get());
  }

  SECTION("works with cuda::stream")
  {
    test_func(stream);
  }

  SECTION("works with cuda::stream_ref")
  {
    test_func(cuda::stream_ref{stream});
  }

  SECTION("works with a type convertible to cudaStream_t")
  {
    test_func(stream_convertible{stream.get()});
  }

  SECTION("works with a non-copyable type convertible to cudaStream_t")
  {
    test_func(stream_convertible_non_copyable{stream.get()});
  }

  SECTION("works with a type with a .stream() accessor")
  {
    test_func(with_stream_method{stream.get()});
  }

  SECTION("works with a type with a .get_stream() accessor")
  {
    test_func(with_get_stream_method{stream.get()});
  }

  SECTION("works with cuda::std::execution::env with cuda::stream_ref")
  {
    test_func(cuda::std::execution::env{cuda::stream_ref{stream}});
  }

  SECTION("works with cuda::std::execution::env with prop with cudaStream_t")
  {
    // MSVC has trouble nesting two aggregate initializations with CTAD
    auto stream_prop = cuda::std::execution::prop{cuda::get_stream, stream.get()};
    test_func(cuda::std::execution::env{cuda::std::move(stream_prop)});
  }

  SECTION("works with cuda::std::execution::env with prop with cuda::stream_ref")
  {
    // MSVC has trouble nesting two aggregate initializations with CTAD
    auto stream_prop = cuda::std::execution::prop{cuda::get_stream, cuda::stream_ref{stream}};
    test_func(cuda::std::execution::env{cuda::std::move(stream_prop)});
  }

  SECTION("works with cuda::execution::gpu with stream")
  {
    test_func(cuda::execution::gpu.with(cuda::get_stream, stream));
  }
}

//! Calls test_func with different versions of a stream, including environments without a stream
template <typename TestFunc>
void test_with_custom_streams(TestFunc&& test_func)
{
  const cuda::stream stream = c2h::make_current_device_stream();
  test_with_custom_streams(test_func, stream);

  SECTION("works with cuda::std::execution::env (no custom stream)")
  {
    const cuda::std::execution::env env{};
    test_func(env);
  }

  SECTION("works with cuda::execution::gpu (no custom stream)")
  {
    const auto policy = cuda::execution::gpu;
    test_func(policy);
  }
}
