//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/buffer>
#include <cuda/memory_resource>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

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
constexpr int get_property(const cuda::experimental::uninitialized_buffer<int, my_property>&, my_property)
{
  return 42;
}

TEMPLATE_TEST_CASE(
  "uninitialized_buffer", "[container]", char, short, int, long, long long, float, double, do_not_construct)
{
  using uninitialized_buffer = cuda::experimental::uninitialized_buffer<TestType>;
  static_assert(!cuda::std::is_default_constructible<uninitialized_buffer>::value, "");
  static_assert(!cuda::std::is_copy_constructible<uninitialized_buffer>::value, "");
  static_assert(!cuda::std::is_copy_assignable<uninitialized_buffer>::value, "");

  cuda::mr::cuda_memory_resource resource{};

  SECTION("construction")
  {
    {
      uninitialized_buffer from_count{resource, 42};
      CHECK(from_count.data() != nullptr);
      CHECK(from_count.size() == 42);
    }
    {
      uninitialized_buffer input{resource, 42};
      const TestType* ptr = input.data();

      uninitialized_buffer from_rvalue{cuda::std::move(input)};
      CHECK(from_rvalue.data() == ptr);
      CHECK(from_rvalue.size() == 42);

      // Ensure that we properly reset the input buffer
      CHECK(input.data() == nullptr);
      CHECK(input.size() == 0);
    }
  }

  SECTION("access")
  {
    uninitialized_buffer buf{resource, 42};
    CHECK(buf.data() != nullptr);
    CHECK(buf.size() == 42);
    CHECK(buf.begin() == buf.data());
    CHECK(buf.end() == buf.begin() + buf.size());
    CHECK(buf.resource() == resource);

    CHECK(cuda::std::as_const(buf).data() != nullptr);
    CHECK(cuda::std::as_const(buf).size() == 42);
    CHECK(cuda::std::as_const(buf).begin() == buf.data());
    CHECK(cuda::std::as_const(buf).end() == buf.begin() + buf.size());
    CHECK(cuda::std::as_const(buf).resource() == resource);
  }

  SECTION("properties")
  {
    static_assert(cuda::has_property<cuda::experimental::uninitialized_buffer<int, cuda::mr::device_accessible>,
                                     cuda::mr::device_accessible>,
                  "");
    static_assert(cuda::has_property<cuda::experimental::uninitialized_buffer<int, my_property>, my_property>, "");
  }

  SECTION("convertion to span")
  {
    uninitialized_buffer buf{resource, 42};
    const cuda::std::span<TestType> as_span{buf};
    CHECK(as_span.data() == buf.data());
    CHECK(as_span.size() == 42);
  }
}
