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

TEMPLATE_TEST_CASE("cudax::vector capacity",
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

  SECTION("cudax::vector::empty")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Vector&>().empty()), bool>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Vector&>().empty()), bool>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Vector&>().empty()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Vector&>().empty()));

    { // Works without allocation
      Vector vec{resource, 0};
      CHECK(vec.empty());
      CHECK(cuda::std::as_const(vec).empty());
    }

    { // Works with allocation and after clear
      Vector vec{resource, 42, cudax::uninit}; // Note we do not care about the elements just the sizes
      CHECK(!vec.empty());
      CHECK(!cuda::std::as_const(vec).empty());

      vec.clear();
      CHECK(vec.empty());
      CHECK(cuda::std::as_const(vec).empty());
    }
  }

  SECTION("cudax::vector::size")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Vector&>().size()), size_type>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Vector&>().size()), size_type>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Vector&>().size()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Vector&>().size()));

    { // Works without allocation
      Vector vec{resource, 0};
      CHECK(vec.size() == 0);
      CHECK(cuda::std::as_const(vec).size() == 0);
    }

    { // Works with allocation and after clear
      Vector vec{resource, 42, cudax::uninit}; // Note we do not care about the elements just the sizes
      CHECK(vec.size() == 42);
      CHECK(cuda::std::as_const(vec).size() == 42);

      vec.clear();
      CHECK(vec.size() == 0);
      CHECK(cuda::std::as_const(vec).size() == 0);
    }
  }

  SECTION("cudax::vector::capacity")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Vector&>().capacity()), size_type>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Vector&>().capacity()), size_type>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Vector&>().capacity()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Vector&>().capacity()));

    { // Works without allocation
      Vector vec{resource, 0};
      CHECK(vec.capacity() == 0);
      CHECK(cuda::std::as_const(vec).capacity() == 0);
    }

    { // Works with allocation and does noth change from clear
      Vector vec{resource, 42, cudax::uninit}; // Note we do not care about the elements just the sizes
      CHECK(vec.capacity() == 42);
      CHECK(cuda::std::as_const(vec).capacity() == 42);

      vec.clear();
      CHECK(vec.capacity() == 42);
      CHECK(cuda::std::as_const(vec).capacity() == 42);
    }
  }

  SECTION("cudax::vector::max_size")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Vector&>().max_size()), size_type>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Vector&>().max_size()), size_type>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Vector&>().max_size()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Vector&>().max_size()));

    constexpr size_t max_size =
      static_cast<size_type>((cuda::std::numeric_limits<typename Vector::difference_type>::max)());
    { // Works without allocation
      Vector vec{resource, 0};
      CHECK(vec.max_size() == max_size);
      CHECK(cuda::std::as_const(vec).max_size() == max_size);
    }

    { // Works with allocation and does noth change from clear
      Vector vec{resource, 42, cudax::uninit}; // Note we do not care about the elements just the sizes
      CHECK(vec.max_size() == max_size);
      CHECK(cuda::std::as_const(vec).max_size() == max_size);

      vec.clear();
      CHECK(vec.max_size() == max_size);
      CHECK(cuda::std::as_const(vec).max_size() == max_size);
    }
  }
}
