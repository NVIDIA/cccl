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

#include "testing.cuh"
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
      CUDAX_CHECK(from_stream_count.data() != nullptr);
      CUDAX_CHECK(from_stream_count.size() == 42);
    }

    {
      uninitialized_async_buffer input{resource, stream, 42};
      const TestType* ptr = input.data();

      uninitialized_async_buffer from_rvalue{cuda::std::move(input)};
      CUDAX_CHECK(from_rvalue.data() == ptr);
      CUDAX_CHECK(from_rvalue.size() == 42);
      CUDAX_CHECK(from_rvalue.get_stream() == stream);

      // Ensure that we properly reset the input buffer
      CUDAX_CHECK(input.data() == nullptr);
      CUDAX_CHECK(input.size() == 0);
      CUDAX_CHECK(input.get_stream() == cuda::stream_ref{});
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
      CUDAX_CHECK(assign_rvalue.data() == ptr);
      CUDAX_CHECK(assign_rvalue.size() == 42);
      CUDAX_CHECK(assign_rvalue.get_stream() == other_stream);

      // Ensure that we properly reset the input buffer
      CUDAX_CHECK(input.data() == nullptr);
      CUDAX_CHECK(input.size() == 0);
      CUDAX_CHECK(input.get_stream() == cuda::stream_ref{});
    }

    { // Ensure self move assignment doesnt do anything
      uninitialized_async_buffer buf{resource, stream, 42};
      const auto* old_ptr = buf.data();

      buf = cuda::std::move(buf);
      CUDAX_CHECK(buf.data() == old_ptr);
      CUDAX_CHECK(buf.get_stream() == stream);
      CUDAX_CHECK(buf.size() == 42);
    }
  }

  SECTION("access")
  {
    uninitialized_async_buffer buf{resource, stream, 42};
    CUDAX_CHECK(buf.data() != nullptr);
    CUDAX_CHECK(buf.size() == 42);
    CUDAX_CHECK(buf.begin() == buf.data());
    CUDAX_CHECK(buf.end() == buf.begin() + buf.size());
    CUDAX_CHECK(buf.get_stream() == stream);
    CUDAX_CHECK(buf.get_resource() == resource);

    CUDAX_CHECK(cuda::std::as_const(buf).data() != nullptr);
    CUDAX_CHECK(cuda::std::as_const(buf).size() == 42);
    CUDAX_CHECK(cuda::std::as_const(buf).begin() == buf.data());
    CUDAX_CHECK(cuda::std::as_const(buf).end() == buf.begin() + buf.size());
    CUDAX_CHECK(cuda::std::as_const(buf).get_stream() == stream);
    CUDAX_CHECK(cuda::std::as_const(buf).get_resource() == resource);
  }

  SECTION("properties")
  {
    static_assert(cuda::has_property<cuda::experimental::uninitialized_async_buffer<int, cuda::mr::device_accessible>,
                                     cuda::mr::device_accessible>,
                  "");
    static_assert(cuda::has_property<cuda::experimental::uninitialized_async_buffer<int, my_property>, my_property>,
                  "");
  }

  SECTION("conversion to span")
  {
    uninitialized_async_buffer buf{resource, stream, 42};
    const cuda::std::span<TestType> as_span{buf};
    CUDAX_CHECK(as_span.data() == buf.data());
    CUDAX_CHECK(as_span.size() == 42);
  }

  SECTION("Actually use memory")
  {
    if constexpr (!cuda::std::is_same_v<TestType, do_not_construct>)
    {
      uninitialized_async_buffer buf{resource, stream, 42};
      stream.wait();
      thrust::fill(thrust::device, buf.begin(), buf.end(), TestType{2});
      const auto res = thrust::reduce(thrust::device, buf.begin(), buf.end(), TestType{0}, thrust::plus<int>());
      CUDAX_CHECK(res == TestType{84});
    }
  }
}

// A test resource that keeps track of the number of resources are
// currently alive.
struct test_async_memory_resource : cudax::mr::async_memory_resource
{
  static int count;

  test_async_memory_resource()
  {
    ++count;
  }

  test_async_memory_resource(const test_async_memory_resource& other)
      : cudax::mr::async_memory_resource{other}
  {
    ++count;
  }

  ~test_async_memory_resource()
  {
    --count;
  }
};

int test_async_memory_resource::count = 0;

TEST_CASE("uninitialized_async_buffer's memory resource does not dangle", "[container]")
{
  cuda::experimental::stream stream{};
  cudax::uninitialized_async_buffer<int, ::cuda::mr::device_accessible> buffer{
    cudax::mr::async_memory_resource{}, stream, 0};

  {
    CHECK(test_async_memory_resource::count == 0);

    cudax::uninitialized_async_buffer<int, ::cuda::mr::device_accessible> src_buffer{
      test_async_memory_resource{}, stream, 1024};

    CHECK(test_async_memory_resource::count == 1);

    cudax::uninitialized_async_buffer<int, ::cuda::mr::device_accessible> dst_buffer{
      src_buffer.get_resource(), stream, 1024};

    CHECK(test_async_memory_resource::count == 2);

    buffer = ::cuda::std::move(dst_buffer);
  }

  CHECK(test_async_memory_resource::count == 1);
}
