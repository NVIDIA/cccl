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
#include <cuda/stream>

#include <cuda/experimental/container.cuh>
#include <cuda/experimental/memory_resource.cuh>

#include "testing.cuh"

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
constexpr int get_property(
  const cuda::experimental::uninitialized_async_buffer<int, cuda::mr::device_accessible, my_property>&, my_property)
{
  return 42;
}
constexpr int get_property(const cuda::device_memory_pool_ref&, my_property)
{
  return 42;
}

C2H_TEST_LIST(
  "uninitialized_async_buffer", "[container]", char, short, int, long, long long, float, double, do_not_construct)
{
  using uninitialized_async_buffer =
    cuda::experimental::uninitialized_async_buffer<TestType, cuda::mr::device_accessible>;
  static_assert(!cuda::std::is_default_constructible<uninitialized_async_buffer>::value, "");
  static_assert(!cuda::std::is_copy_constructible<uninitialized_async_buffer>::value, "");
  static_assert(!cuda::std::is_copy_assignable<uninitialized_async_buffer>::value, "");

  cuda::device_memory_pool_ref resource = cuda::device_default_memory_pool(cuda::device_ref{0});
  cuda::stream stream{cuda::device_ref{0}};

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
      CUDAX_CHECK(from_rvalue.stream() == stream);

      // Ensure that we properly reset the input buffer
      CUDAX_CHECK(input.data() == nullptr);
      CUDAX_CHECK(input.size() == 0);
      CUDAX_CHECK(input.stream() == cuda::stream_ref{cudaStream_t{}});
    }
  }

  SECTION("conversion")
  {
    cuda::experimental::uninitialized_async_buffer<TestType, cuda::mr::device_accessible, my_property> input{
      resource, stream, 42};
    const TestType* ptr = input.data();

    uninitialized_async_buffer from_rvalue{cuda::std::move(input)};
    CUDAX_CHECK(from_rvalue.data() == ptr);
    CUDAX_CHECK(from_rvalue.size() == 42);
    CUDAX_CHECK(from_rvalue.stream() == stream);

    // Ensure that we properly reset the input buffer
    CUDAX_CHECK(input.data() == nullptr);
    CUDAX_CHECK(input.size() == 0);
    CUDAX_CHECK(input.stream() == cuda::stream_ref{cudaStream_t{}});
  }

  SECTION("assignment")
  {
    static_assert(!cuda::std::is_copy_assignable<uninitialized_async_buffer>::value, "");

    {
      cuda::stream other_stream{cuda::device_ref{0}};
      uninitialized_async_buffer input{resource, other_stream, 42};
      const TestType* ptr = input.data();

      uninitialized_async_buffer assign_rvalue{resource, stream, 1337};
      assign_rvalue = cuda::std::move(input);
      CUDAX_CHECK(assign_rvalue.data() == ptr);
      CUDAX_CHECK(assign_rvalue.size() == 42);
      CUDAX_CHECK(assign_rvalue.size_bytes() == 42 * sizeof(TestType));
      CUDAX_CHECK(assign_rvalue.stream() == other_stream);

      // Ensure that we properly reset the input buffer
      CUDAX_CHECK(input.data() == nullptr);
      CUDAX_CHECK(input.size() == 0);
      CUDAX_CHECK(input.size_bytes() == 0);
      CUDAX_CHECK(input.stream() == cuda::stream_ref{cudaStream_t{}});
    }

    { // Ensure self move assignment does not do anything
      uninitialized_async_buffer buf{resource, stream, 42};
      const auto* old_ptr = buf.data();

      buf = cuda::std::move(buf);
      CUDAX_CHECK(buf.data() == old_ptr);
      CUDAX_CHECK(buf.stream() == stream);
      CUDAX_CHECK(buf.size() == 42);
      CUDAX_CHECK(buf.size_bytes() == 42 * sizeof(TestType));
    }
  }

  SECTION("access")
  {
    uninitialized_async_buffer buf{resource, stream, 42};
    static_assert(cuda::std::is_same<decltype(buf.begin()), TestType*>::value, "");
    static_assert(cuda::std::is_same<decltype(buf.end()), TestType*>::value, "");
    static_assert(cuda::std::is_same<decltype(buf.data()), TestType*>::value, "");
    CUDAX_CHECK(buf.data() != nullptr);
    CUDAX_CHECK(buf.size() == 42);
    CUDAX_CHECK(buf.size_bytes() == 42 * sizeof(TestType));
    CUDAX_CHECK(buf.begin() == buf.data());
    CUDAX_CHECK(buf.end() == buf.begin() + buf.size());
    CUDAX_CHECK(buf.stream() == stream);
    CUDAX_CHECK(buf.memory_resource() == resource);

    static_assert(cuda::std::is_same<decltype(cuda::std::as_const(buf).begin()), TestType const*>::value, "");
    static_assert(cuda::std::is_same<decltype(cuda::std::as_const(buf).end()), TestType const*>::value, "");
    static_assert(cuda::std::is_same<decltype(cuda::std::as_const(buf).data()), TestType const*>::value, "");
    CUDAX_CHECK(cuda::std::as_const(buf).data() != nullptr);
    CUDAX_CHECK(cuda::std::as_const(buf).size() == 42);
    CUDAX_CHECK(cuda::std::as_const(buf).size_bytes() == 42 * sizeof(TestType));
    CUDAX_CHECK(cuda::std::as_const(buf).begin() == buf.data());
    CUDAX_CHECK(cuda::std::as_const(buf).end() == buf.begin() + buf.size());
    CUDAX_CHECK(cuda::std::as_const(buf).stream() == stream);
    CUDAX_CHECK(cuda::std::as_const(buf).memory_resource() == resource);
  }

  SECTION("properties")
  {
    static_assert(cuda::has_property<cuda::experimental::uninitialized_async_buffer<int, cuda::mr::device_accessible>,
                                     cuda::mr::device_accessible>,
                  "");
    static_assert(
      cuda::has_property<cuda::experimental::uninitialized_async_buffer<int, cuda::mr::device_accessible, my_property>,
                         my_property>,
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
      stream.sync();
      thrust::fill(thrust::device, buf.begin(), buf.end(), TestType{2});
      const auto res = thrust::reduce(thrust::device, buf.begin(), buf.end(), TestType{0}, cuda::std::plus<int>());
      CUDAX_CHECK(res == TestType{84});
    }
  }

  SECTION("Replace allocation of current buffer")
  {
    uninitialized_async_buffer buf{resource, stream, 42};
    const TestType* old_ptr = buf.data();
    const size_t old_size   = buf.size();

    {
      const uninitialized_async_buffer old_buf = buf.__replace_allocation(1337);
      CUDAX_CHECK(buf.data() != old_ptr);
      CUDAX_CHECK(buf.size() == 1337);

      CUDAX_CHECK(old_buf.data() == old_ptr);
      CUDAX_CHECK(old_buf.size() == old_size);

      CUDAX_CHECK(buf.stream() == old_buf.stream());
    }
  }

  SECTION("destroy")
  {
    uninitialized_async_buffer buf{resource, stream, 42};
    buf.destroy();
    CUDAX_CHECK(buf.data() == nullptr);
    CUDAX_CHECK(buf.size() == 0);
    CUDAX_CHECK(buf.stream() == stream);

    buf = uninitialized_async_buffer{resource, stream, 42};
    CUDAX_CHECK(buf.data() != nullptr);
    CUDAX_CHECK(buf.size() == 42);
    CUDAX_CHECK(buf.stream() == stream);
  }
}

// A test resource that keeps track of the number of resources are
// currently alive.
struct test_async_device_memory_pool_ref : cuda::device_memory_pool_ref
{
  static int count;

  test_async_device_memory_pool_ref()
      : cuda::device_memory_pool_ref(cuda::device_default_memory_pool(cuda::device_ref{0}))
  {
    ++count;
  }

  test_async_device_memory_pool_ref(const test_async_device_memory_pool_ref& other)
      : cuda::device_memory_pool_ref{other}
  {
    ++count;
  }

  ~test_async_device_memory_pool_ref()
  {
    --count;
  }
};

int test_async_device_memory_pool_ref::count = 0;

C2H_TEST("uninitialized_async_buffer's memory resource does not dangle", "[container]")
{
  cuda::stream stream{cuda::device_ref{0}};
  cudax::uninitialized_async_buffer<int, ::cuda::mr::device_accessible> buffer{
    cuda::device_default_memory_pool(cuda::device_ref{0}), stream, 0};

  {
    CHECK(test_async_device_memory_pool_ref::count == 0);

    cudax::uninitialized_async_buffer<int, ::cuda::mr::device_accessible> src_buffer{
      test_async_device_memory_pool_ref{}, stream, 1024};

    CHECK(test_async_device_memory_pool_ref::count == 1);

    cudax::uninitialized_async_buffer<int, ::cuda::mr::device_accessible> dst_buffer{
      src_buffer.memory_resource(), stream, 1024};

    CHECK(test_async_device_memory_pool_ref::count == 2);

    buffer = ::cuda::std::move(dst_buffer);
  }

  CHECK(test_async_device_memory_pool_ref::count == 1);
}
