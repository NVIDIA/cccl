//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

#include <cuda/__container/uninitialized_async_buffer.h>
#include <cuda/memory_pool>
#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <test_resources.h>

#include "testing.cuh"

#if _CCCL_COMPILER(GCC, >=, 13)
_CCCL_DIAG_SUPPRESS_GCC("-Wself-move")
#endif // _CCCL_COMPILER(GCC, >=, 13)
_CCCL_DIAG_SUPPRESS_CLANG("-Wself-move")

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
constexpr int get_property(const cuda::__uninitialized_async_buffer<int, cuda::mr::device_accessible, my_property>&,
                           my_property)
{
  return 42;
}
constexpr int get_property(const cuda::device_memory_pool_ref&, my_property)
{
  return 42;
}

C2H_TEST_LIST(
  "__uninitialized_async_buffer", "[container]", char, short, int, long, long long, float, double, do_not_construct)
{
  using __uninitialized_async_buffer = cuda::__uninitialized_async_buffer<TestType, cuda::mr::device_accessible>;
  static_assert(!cuda::std::is_default_constructible<__uninitialized_async_buffer>::value);
  static_assert(!cuda::std::is_copy_constructible<__uninitialized_async_buffer>::value);
  static_assert(!cuda::std::is_copy_assignable<__uninitialized_async_buffer>::value);

  cuda::device_memory_pool_ref resource = cuda::device_default_memory_pool(cuda::device_ref{0});
  cuda::stream stream{cuda::device_ref{0}};

  SECTION("construction")
  {
    {
      __uninitialized_async_buffer from_stream_count{resource, stream, 42};
      CHECK(from_stream_count.data() != nullptr);
      CHECK(from_stream_count.size() == 42);
    }

    {
      const ::cuda::std::size_t alignment = 64;
      offset_by_alignment_resource aligned_resource{resource};
      __uninitialized_async_buffer from_stream_count{aligned_resource, stream, 42, alignment};
      CHECK(is_pointer_aligned(from_stream_count.data(), alignment));
      CHECK(from_stream_count.alignment() == alignment);
    }

    {
      __uninitialized_async_buffer input{resource, stream, 42};
      const TestType* ptr = input.data();

      __uninitialized_async_buffer from_rvalue{cuda::std::move(input)};
      CHECK(from_rvalue.data() == ptr);
      CHECK(from_rvalue.size() == 42);
      CHECK(from_rvalue.stream() == stream);

      // Ensure that we properly reset the input buffer
      CHECK(input.data() == nullptr);
      CHECK(input.size() == 0);
      CHECK(input.stream() == cuda::stream_ref{cudaStream_t{}});
    }
  }

  SECTION("conversion")
  {
    cuda::__uninitialized_async_buffer<TestType, cuda::mr::device_accessible, my_property> input{resource, stream, 42};
    const TestType* ptr = input.data();

    __uninitialized_async_buffer from_rvalue{cuda::std::move(input)};
    CHECK(from_rvalue.data() == ptr);
    CHECK(from_rvalue.size() == 42);
    CHECK(from_rvalue.stream() == stream);

    // Ensure that we properly reset the input buffer
    CHECK(input.data() == nullptr);
    CHECK(input.size() == 0);
    CHECK(input.stream() == cuda::stream_ref{cudaStream_t{}});
  }

  SECTION("assignment")
  {
    static_assert(!cuda::std::is_copy_assignable<__uninitialized_async_buffer>::value);

    {
      cuda::stream other_stream{cuda::device_ref{0}};
      __uninitialized_async_buffer input{resource, other_stream, 42};
      const TestType* ptr = input.data();

      __uninitialized_async_buffer assign_rvalue{resource, stream, 1337};
      assign_rvalue = cuda::std::move(input);
      CHECK(assign_rvalue.data() == ptr);
      CHECK(assign_rvalue.size() == 42);
      CHECK(assign_rvalue.size_bytes() == 42 * sizeof(TestType));
      CHECK(assign_rvalue.stream() == other_stream);

      // Ensure that we properly reset the input buffer
      CHECK(input.data() == nullptr);
      CHECK(input.size() == 0);
      CHECK(input.size_bytes() == 0);
      CHECK(input.stream() == cuda::stream_ref{cudaStream_t{}});
    }

    { // Ensure self move assignment does not do anything
      __uninitialized_async_buffer buf{resource, stream, 42};
      const auto* old_ptr = buf.data();

      buf = cuda::std::move(buf);
      CHECK(buf.data() == old_ptr);
      CHECK(buf.stream() == stream);
      CHECK(buf.size() == 42);
      CHECK(buf.size_bytes() == 42 * sizeof(TestType));
    }
  }

  SECTION("access")
  {
    __uninitialized_async_buffer buf{resource, stream, 42};
    static_assert(cuda::std::is_same<decltype(buf.begin()), TestType*>::value);
    static_assert(cuda::std::is_same<decltype(buf.end()), TestType*>::value);
    static_assert(cuda::std::is_same<decltype(buf.data()), TestType*>::value);
    CHECK(buf.data() != nullptr);
    CHECK(buf.size() == 42);
    CHECK(buf.size_bytes() == 42 * sizeof(TestType));
    CHECK(buf.begin() == buf.data());
    CHECK(buf.end() == buf.begin() + buf.size());
    CHECK(buf.stream() == stream);
    CHECK(buf.memory_resource() == resource);

    static_assert(cuda::std::is_same<decltype(cuda::std::as_const(buf).begin()), TestType const*>::value);
    static_assert(cuda::std::is_same<decltype(cuda::std::as_const(buf).end()), TestType const*>::value);
    static_assert(cuda::std::is_same<decltype(cuda::std::as_const(buf).data()), TestType const*>::value);
    CHECK(cuda::std::as_const(buf).data() != nullptr);
    CHECK(cuda::std::as_const(buf).size() == 42);
    CHECK(cuda::std::as_const(buf).size_bytes() == 42 * sizeof(TestType));
    CHECK(cuda::std::as_const(buf).begin() == buf.data());
    CHECK(cuda::std::as_const(buf).end() == buf.begin() + buf.size());
    CHECK(cuda::std::as_const(buf).stream() == stream);
    CHECK(cuda::std::as_const(buf).memory_resource() == resource);
  }

  SECTION("properties")
  {
    static_assert(cuda::has_property<cuda::__uninitialized_async_buffer<int, cuda::mr::device_accessible>,
                                     cuda::mr::device_accessible>);
    static_assert(cuda::has_property<cuda::__uninitialized_async_buffer<int, cuda::mr::device_accessible, my_property>,
                                     my_property>);
  }

  SECTION("conversion to span")
  {
    __uninitialized_async_buffer buf{resource, stream, 42};
    const cuda::std::span<TestType> as_span{buf};
    CHECK(as_span.data() == buf.data());
    CHECK(as_span.size() == 42);
  }

  SECTION("Actually use memory")
  {
    if constexpr (!cuda::std::is_same_v<TestType, do_not_construct>)
    {
      __uninitialized_async_buffer buf{resource, stream, 42};
      stream.sync();
      thrust::fill(thrust::device, buf.begin(), buf.end(), TestType{2});
      const auto res = thrust::reduce(thrust::device, buf.begin(), buf.end(), TestType{0}, cuda::std::plus<int>());
      CHECK(res == TestType{84});
    }
  }

  SECTION("Replace allocation of current buffer")
  {
    __uninitialized_async_buffer buf{resource, stream, 42};
    const TestType* old_ptr = buf.data();
    const size_t old_size   = buf.size();

    {
      const __uninitialized_async_buffer old_buf = buf.__replace_allocation(1337);
      CHECK(buf.data() != old_ptr);
      CHECK(buf.size() == 1337);

      CHECK(old_buf.data() == old_ptr);
      CHECK(old_buf.size() == old_size);

      CHECK(buf.stream() == old_buf.stream());
    }
  }

  SECTION("destroy")
  {
    __uninitialized_async_buffer buf{resource, stream, 42};
    buf.destroy();
    CHECK(buf.data() == nullptr);
    CHECK(buf.size() == 0);
    CHECK(buf.stream() == stream);

    buf = __uninitialized_async_buffer{resource, stream, 42};
    CHECK(buf.data() != nullptr);
    CHECK(buf.size() == 42);
    CHECK(buf.stream() == stream);
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

C2H_TEST("__uninitialized_async_buffer's memory resource does not dangle", "[container]")
{
  cuda::stream stream{cuda::device_ref{0}};
  cuda::__uninitialized_async_buffer<int, ::cuda::mr::device_accessible> buffer{
    cuda::device_default_memory_pool(cuda::device_ref{0}), stream, 0};

  {
    CHECK(test_async_device_memory_pool_ref::count == 0);

    cuda::__uninitialized_async_buffer<int, ::cuda::mr::device_accessible> src_buffer{
      test_async_device_memory_pool_ref{}, stream, 1024};

    CHECK(test_async_device_memory_pool_ref::count == 1);

    cuda::__uninitialized_async_buffer<int, ::cuda::mr::device_accessible> dst_buffer{
      src_buffer.memory_resource(), stream, 1024};

    CHECK(test_async_device_memory_pool_ref::count == 2);

    buffer = ::cuda::std::move(dst_buffer);
  }

  CHECK(test_async_device_memory_pool_ref::count == 1);
}
