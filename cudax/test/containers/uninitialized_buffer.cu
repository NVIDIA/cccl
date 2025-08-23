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

#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/container.cuh>
#include <cuda/experimental/launch.cuh>
#include <cuda/experimental/memory_resource.cuh>
#include <cuda/experimental/stream.cuh>

#include "testing.cuh"

struct do_not_construct
{
  do_not_construct()
  {
    CUDAX_CHECK(false);
  }
};

struct non_trivial
{
  int val_      = 0;
  non_trivial() = default;
  __host__ __device__ constexpr non_trivial(const int val) noexcept
      : val_(val)
  {}

  __host__ __device__ constexpr friend bool operator==(const non_trivial& lhs, const non_trivial& rhs)
  {
    return lhs.val_ == rhs.val_;
  }
};

struct my_property
{
  using value_type = int;
};
constexpr int get_property(
  const cuda::experimental::uninitialized_buffer<int, cuda::mr::device_accessible, my_property>&, my_property)
{
  return 42;
}
constexpr int get_property(const cudax::device_memory_resource&, my_property)
{
  return 42;
}

__global__ void kernel(::cuda::std::span<int> data)
{
  // Touch the memory to be sure it's accessible
  CUDAX_CHECK(data.size() == 1024);
  data[0] = 42;
}

__global__ void const_kernel(::cuda::std::span<const int> data)
{
  // Touch the memory to be sure it's accessible
  CUDAX_CHECK(data.size() == 1024);
}

C2H_TEST_LIST("uninitialized_buffer", "[container]", char, short, int, long, long long, float, double, do_not_construct)
{
  using uninitialized_buffer = cuda::experimental::uninitialized_buffer<TestType, cuda::mr::device_accessible>;
  static_assert(!cuda::std::is_default_constructible<uninitialized_buffer>::value, "");
  static_assert(!cuda::std::is_copy_constructible<uninitialized_buffer>::value, "");
  static_assert(!cuda::std::is_copy_assignable<uninitialized_buffer>::value, "");

  cudax::device_memory_resource resource{cuda::device_ref{0}};

  SECTION("construction")
  {
    static_assert(!cuda::std::is_copy_constructible<uninitialized_buffer>::value, "");
    {
      uninitialized_buffer from_count{resource, 42};
      CUDAX_CHECK(from_count.data() != nullptr);
      CUDAX_CHECK(from_count.size() == 42);
    }
    {
      uninitialized_buffer input{resource, 42};
      const TestType* ptr = input.data();

      uninitialized_buffer from_rvalue{cuda::std::move(input)};
      CUDAX_CHECK(from_rvalue.data() == ptr);
      CUDAX_CHECK(from_rvalue.size() == 42);

      // Ensure that we properly reset the input buffer
      CUDAX_CHECK(input.data() == nullptr);
      CUDAX_CHECK(input.size() == 0);
    }
  }

  SECTION("conversion")
  {
    cuda::experimental::uninitialized_buffer<TestType, cuda::mr::device_accessible, my_property> input{resource, 42};
    const TestType* ptr = input.data();

    uninitialized_buffer from_rvalue{cuda::std::move(input)};
    CUDAX_CHECK(from_rvalue.data() == ptr);
    CUDAX_CHECK(from_rvalue.size() == 42);

    // Ensure that we properly reset the input buffer
    CUDAX_CHECK(input.data() == nullptr);
    CUDAX_CHECK(input.size() == 0);
  }

  SECTION("assignment")
  {
    static_assert(!cuda::std::is_copy_assignable<uninitialized_buffer>::value, "");
    {
      cudax::legacy_pinned_memory_resource other_resource{};
      uninitialized_buffer input{other_resource, 42};
      uninitialized_buffer buf{resource, 1337};
      const auto* old_ptr       = buf.data();
      const auto* old_input_ptr = input.data();

      buf = cuda::std::move(input);
      CUDAX_CHECK(buf.data() != old_ptr);
      CUDAX_CHECK(buf.data() == old_input_ptr);
      CUDAX_CHECK(buf.size() == 42);
      CUDAX_CHECK(buf.size_bytes() == 42 * sizeof(TestType));
      CUDAX_CHECK(buf.memory_resource() == other_resource);

      CUDAX_CHECK(input.data() == nullptr);
      CUDAX_CHECK(input.size() == 0);
      CUDAX_CHECK(input.size_bytes() == 0);
    }

    { // Ensure self move assignment does not do anything
      uninitialized_buffer buf{resource, 1337};
      const auto* old_ptr = buf.data();

      buf = cuda::std::move(buf);
      CUDAX_CHECK(buf.data() == old_ptr);
      CUDAX_CHECK(buf.size() == 1337);
      CUDAX_CHECK(buf.size_bytes() == 1337 * sizeof(TestType));
    }
  }

  SECTION("access")
  {
    uninitialized_buffer buf{resource, 42};
    static_assert(cuda::std::is_same<decltype(buf.begin()), TestType*>::value, "");
    static_assert(cuda::std::is_same<decltype(buf.end()), TestType*>::value, "");
    static_assert(cuda::std::is_same<decltype(buf.data()), TestType*>::value, "");
    CUDAX_CHECK(buf.data() != nullptr);
    CUDAX_CHECK(buf.size() == 42);
    CUDAX_CHECK(buf.size_bytes() == 42 * sizeof(TestType));
    CUDAX_CHECK(buf.begin() == buf.data());
    CUDAX_CHECK(buf.end() == buf.begin() + buf.size());
    CUDAX_CHECK(buf.memory_resource() == resource);

    static_assert(cuda::std::is_same<decltype(cuda::std::as_const(buf).begin()), TestType const*>::value, "");
    static_assert(cuda::std::is_same<decltype(cuda::std::as_const(buf).end()), TestType const*>::value, "");
    static_assert(cuda::std::is_same<decltype(cuda::std::as_const(buf).data()), TestType const*>::value, "");
    CUDAX_CHECK(cuda::std::as_const(buf).data() != nullptr);
    CUDAX_CHECK(cuda::std::as_const(buf).size() == 42);
    CUDAX_CHECK(cuda::std::as_const(buf).begin() == buf.data());
    CUDAX_CHECK(cuda::std::as_const(buf).end() == buf.begin() + buf.size());
    CUDAX_CHECK(cuda::std::as_const(buf).memory_resource() == resource);
  }

  SECTION("properties")
  {
    static_assert(cuda::has_property<cuda::experimental::uninitialized_buffer<int, cuda::mr::device_accessible>,
                                     cuda::mr::device_accessible>,
                  "");
    static_assert(
      cuda::has_property<cuda::experimental::uninitialized_buffer<int, cuda::mr::device_accessible, my_property>,
                         my_property>,
      "");
  }

  SECTION("conversion to span")
  {
    uninitialized_buffer buf{resource, 42};
    const cuda::std::span<TestType> as_span{buf};
    CUDAX_CHECK(as_span.data() == buf.data());
    CUDAX_CHECK(as_span.size() == 42);
  }

  SECTION("Actually use memory")
  {
    if constexpr (!cuda::std::is_same_v<TestType, do_not_construct>)
    {
      uninitialized_buffer buf{resource, 42};
      thrust::fill(thrust::device, buf.begin(), buf.end(), TestType{2});
      const auto res = thrust::reduce(thrust::device, buf.begin(), buf.end(), TestType{0}, cuda::std::plus<int>());
      CUDAX_CHECK(res == TestType{84});
    }
  }

  SECTION("Replace allocation of current buffer")
  {
    uninitialized_buffer buf{resource, 42};
    const TestType* old_ptr = buf.data();
    const size_t old_size   = buf.size();

    {
      const uninitialized_buffer old_buf = buf.__replace_allocation(1337);
      CUDAX_CHECK(buf.data() != old_ptr);
      CUDAX_CHECK(buf.size() == 1337);

      CUDAX_CHECK(old_buf.data() == old_ptr);
      CUDAX_CHECK(old_buf.size() == old_size);
    }
  }

  SECTION("destroy")
  {
    uninitialized_buffer buf{resource, 42};
    buf.destroy();
    CUDAX_CHECK(buf.data() == nullptr);
    CUDAX_CHECK(buf.size() == 0);

    buf = uninitialized_buffer{resource, 42};
    CUDAX_CHECK(buf.data() != nullptr);
    CUDAX_CHECK(buf.size() == 42);
  }
}

C2H_TEST("uninitialized_buffer is usable with cudax::launch", "[container]")
{
  SECTION("non-const")
  {
    const int grid_size = 4;
    cudax::uninitialized_buffer<int, ::cuda::mr::device_accessible> buffer{
      cudax::device_memory_resource{cuda::device_ref{0}}, 1024};
    auto configuration = cudax::make_config(cudax::grid_dims(grid_size), cudax::block_dims<256>());

    cudax::stream stream{cuda::device_ref{0}};

    cudax::launch(stream, configuration, kernel, buffer);
  }

  SECTION("const")
  {
    const int grid_size = 4;
    const cudax::uninitialized_buffer<int, ::cuda::mr::device_accessible> buffer{
      cudax::device_memory_resource{cuda::device_ref{0}}, 1024};
    auto configuration = cudax::make_config(cudax::grid_dims(grid_size), cudax::block_dims<256>());

    cudax::stream stream{cuda::device_ref{0}};

    cudax::launch(stream, configuration, const_kernel, buffer);
  }
}

// A test resource that keeps track of the number of resources are
// currently alive.
struct test_device_memory_resource : cudax::device_memory_resource
{
  static int count;

  test_device_memory_resource()
      : cudax::device_memory_resource{cuda::device_ref{0}}
  {
    ++count;
  }

  test_device_memory_resource(const test_device_memory_resource& other)
      : cudax::device_memory_resource{other}
  {
    ++count;
  }

  ~test_device_memory_resource()
  {
    --count;
  }
};

int test_device_memory_resource::count = 0;

C2H_TEST("uninitialized_buffer's memory resource does not dangle", "[container]")
{
  cudax::uninitialized_buffer<int, ::cuda::mr::device_accessible> buffer{
    cudax::device_memory_resource{cuda::device_ref{0}}, 0};

  {
    CHECK(test_device_memory_resource::count == 0);

    cudax::uninitialized_buffer<int, ::cuda::mr::device_accessible> src_buffer{test_device_memory_resource{}, 1024};

    CHECK(test_device_memory_resource::count == 1);

    cudax::uninitialized_buffer<int, ::cuda::mr::device_accessible> dst_buffer{src_buffer.memory_resource(), 1024};

    CHECK(test_device_memory_resource::count == 2);

    buffer = ::cuda::std::move(dst_buffer);
  }

  CHECK(test_device_memory_resource::count == 1);
}
