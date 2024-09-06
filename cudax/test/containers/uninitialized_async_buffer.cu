//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream_ref>

#include <cuda/experimental/buffer.cuh>
#include <cuda/experimental/memory_resource.cuh>

#include <catch2/catch.hpp>

struct do_not_construct
{
  do_not_construct()
  {
    CHECK(false);
  }
};

struct my_property
{
  using value_type = int;
};
constexpr int get_property(const cuda::experimental::uninitialized_async_buffer<int, my_property>&, my_property)
{
  return 42;
}

TEMPLATE_TEST_CASE(
  "uninitialized_async_buffer", "[container]", char, short, int, long, long long, float, double, do_not_construct)
{
  using uninitialized_async_buffer = cuda::experimental::uninitialized_async_buffer<TestType>;
  static_assert(!cuda::std::is_default_constructible<uninitialized_async_buffer>::value, "");
  static_assert(!cuda::std::is_copy_constructible<uninitialized_async_buffer>::value, "");
  static_assert(!cuda::std::is_copy_assignable<uninitialized_async_buffer>::value, "");

  cuda::experimental::mr::async_memory_resource resource{};
  cuda::experimental::stream stream{};

  SECTION("construction")
  {
    {
      uninitialized_async_buffer from_stream_count{resource, stream, 42};
      CHECK(from_stream_count.data() != nullptr);
      CHECK(from_stream_count.size() == 42);
    }

    {
      uninitialized_async_buffer input{resource, stream, 42};
      const TestType* ptr = input.data();

      uninitialized_async_buffer from_rvalue{cuda::std::move(input)};
      CHECK(from_rvalue.data() == ptr);
      CHECK(from_rvalue.size() == 42);
      CHECK(from_rvalue.get_stream() == stream);

      // Ensure that we properly reset the input buffer
      CHECK(input.data() == nullptr);
      CHECK(input.size() == 0);
      CHECK(input.get_stream() == cuda::stream_ref{});
    }
  }

  SECTION("assignment")
  {
    static_assert(!cuda::std::is_copy_assignable<uninitialized_async_buffer>::value, "");

    {
      cuda::experimental::stream other_stream{};
      uninitialized_async_buffer input{resource, other_stream, 42};
      const TestType* ptr = input.data();

      uninitialized_async_buffer assign_rvalue{resource, stream, 1337};
      assign_rvalue = cuda::std::move(input);
      CHECK(assign_rvalue.data() == ptr);
      CHECK(assign_rvalue.size() == 42);
      CHECK(assign_rvalue.get_stream() == other_stream);

      // Ensure that we properly reset the input buffer
      CHECK(input.data() == nullptr);
      CHECK(input.size() == 0);
      CHECK(input.get_stream() == cuda::stream_ref{});
    }

    { // Ensure self move assignment doesnt do anything
      uninitialized_async_buffer buf{resource, stream, 42};
      const auto* old_ptr = buf.data();

      buf = cuda::std::move(buf);
      CHECK(buf.data() == old_ptr);
      CHECK(buf.get_stream() == stream);
      CHECK(buf.size() == 42);
    }
  }

  SECTION("access")
  {
    uninitialized_async_buffer buf{resource, stream, 42};
    CHECK(buf.data() != nullptr);
    CHECK(buf.size() == 42);
    CHECK(buf.begin() == buf.data());
    CHECK(buf.end() == buf.begin() + buf.size());
    CHECK(buf.get_stream() == stream);

    CHECK(cuda::std::as_const(buf).data() != nullptr);
    CHECK(cuda::std::as_const(buf).size() == 42);
    CHECK(cuda::std::as_const(buf).begin() == buf.data());
    CHECK(cuda::std::as_const(buf).end() == buf.begin() + buf.size());
    CHECK(cuda::std::as_const(buf).get_stream() == stream);
  }

  SECTION("properties")
  {
    static_assert(cuda::has_property<cuda::experimental::uninitialized_async_buffer<int, cuda::mr::device_accessible>,
                                     cuda::mr::device_accessible>,
                  "");
    static_assert(cuda::has_property<cuda::experimental::uninitialized_async_buffer<int, my_property>, my_property>,
                  "");
  }

  SECTION("convertion to span")
  {
    uninitialized_async_buffer buf{resource, stream, 42};
    const cuda::std::span<TestType> as_span{buf};
    CHECK(as_span.data() == buf.data());
    CHECK(as_span.size() == 42);
  }

  SECTION("Actually use memory")
  {
    if constexpr (!cuda::std::is_same_v<TestType, do_not_construct>)
    {
      uninitialized_async_buffer buf{resource, stream, 42};
      stream.wait();
      thrust::fill(thrust::device, buf.begin(), buf.end(), TestType{2});
      const auto res = thrust::reduce(thrust::device, buf.begin(), buf.end(), TestType{0}, thrust::plus<int>());
      CHECK(res == TestType{84});
    }
  }
}
