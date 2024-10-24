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

#include <cuda/experimental/vector.cuh>

#include "helper.h"
#include "types.h"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE("cudax::vector swap",
                   "[container][vector]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Resource     = typename extract_properties<TestType>::resource;
  using Resource_ref = typename extract_properties<TestType>::resource_ref;
  using Vector       = typename extract_properties<TestType>::vector;
  using T            = typename Vector::value_type;
  using size_type    = typename Vector::size_type;

  Resource raw_resource{};
  Resource_ref resource{raw_resource};
  STATIC_REQUIRE(
    cuda::std::is_same_v<decltype(cuda::std::declval<Vector&>().swap(cuda::std::declval<Vector&>())), void>);
  STATIC_REQUIRE(
    cuda::std::is_same_v<decltype(swap(cuda::std::declval<Vector&>(), cuda::std::declval<Vector&>())), void>);
  STATIC_REQUIRE(noexcept(cuda::std::declval<Vector&>().swap(cuda::std::declval<Vector&>())));
  STATIC_REQUIRE(noexcept(swap(cuda::std::declval<Vector&>(), cuda::std::declval<Vector&>())));

  // Note we do not care about the elements just the sizes
  Vector vec_small{resource, 5, cudax::uninit};

  SECTION("Can swap vector")
  {
    Vector vec_large{resource, 42, cudax::uninit};

    CHECK(vec_large.capacity() == 42);
    CHECK(vec_small.capacity() == 5);
    CHECK(vec_large.size() == 42);
    CHECK(vec_small.size() == 5);

    vec_large.swap(vec_small);
    CHECK(vec_small.capacity() == 42);
    CHECK(vec_large.capacity() == 5);
    CHECK(vec_small.size() == 42);
    CHECK(vec_large.size() == 5);

    swap(vec_large, vec_small);
    CHECK(vec_large.capacity() == 42);
    CHECK(vec_small.capacity() == 5);
    CHECK(vec_large.size() == 42);
    CHECK(vec_small.size() == 5);
  }

  SECTION("Can swap vector without allocation")
  {
    Vector vec_no_allocation{resource, 0, cudax::uninit};

    CHECK(vec_no_allocation.capacity() == 0);
    CHECK(vec_small.capacity() == 5);
    CHECK(vec_no_allocation.size() == 0);
    CHECK(vec_small.size() == 5);

    vec_no_allocation.swap(vec_small);
    CHECK(vec_small.capacity() == 0);
    CHECK(vec_no_allocation.capacity() == 5);
    CHECK(vec_small.size() == 0);
    CHECK(vec_no_allocation.size() == 5);

    swap(vec_no_allocation, vec_small);
    CHECK(vec_no_allocation.capacity() == 0);
    CHECK(vec_small.capacity() == 5);
    CHECK(vec_no_allocation.size() == 0);
    CHECK(vec_small.size() == 5);
  }
}
