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

#include <cuda/experimental/container.cuh>

#include "helper.h"
#include "types.h"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE("cudax::async_mdarray iterators",
                   "[container][async_mdarray]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Env       = typename extract_properties<TestType>::env;
  using Resource  = typename extract_properties<TestType>::resource;
  using Array     = typename extract_properties<TestType>::async_mdarray;
  using T         = typename Array::value_type;
  using size_type = typename Array::size_type;

  using iterator       = typename extract_properties<TestType>::iterator;
  using const_iterator = typename extract_properties<TestType>::const_iterator;

  using reverse_iterator       = cuda::std::reverse_iterator<iterator>;
  using const_reverse_iterator = cuda::std::reverse_iterator<const_iterator>;

  cudax::stream stream{};
  Env env{Resource{}, stream};

  SECTION("cudax::async_mdarray::begin/end properties")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Array&>().begin()), iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Array&>().begin()), const_iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Array&>().cbegin()), const_iterator>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Array&>().begin()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Array&>().begin()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<Array&>().cbegin()));

    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Array&>().end()), iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Array&>().end()), const_iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Array&>().cend()), const_iterator>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Array&>().end()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Array&>().end()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<Array&>().cend()));
  }

  SECTION("cudax::async_mdarray::begin/end no allocation")
  {
    Array vec{env, 0};
    CHECK(vec.begin() == iterator{nullptr});
    CHECK(cuda::std::as_const(vec).begin() == const_iterator{nullptr});
    CHECK(vec.cbegin() == const_iterator{nullptr});

    CHECK(vec.end() == iterator{nullptr});
    CHECK(cuda::std::as_const(vec).end() == const_iterator{nullptr});
    CHECK(vec.cend() == const_iterator{nullptr});

    CHECK(vec.begin() == vec.end());
    CHECK(cuda::std::as_const(vec).begin() == cuda::std::as_const(vec).end());
    CHECK(vec.cbegin() == vec.cend());
  }

  SECTION("cudax::async_mdarray::begin/end with allocation")
  {
    Array vec{env, 42, cudax::uninit}; // Note we do not care about the elements just the sizes
    // begin points to the element at data()
    CHECK(vec.begin() == iterator{vec.data()});
    CHECK(cuda::std::as_const(vec).begin() == const_iterator{vec.data()});
    CHECK(vec.cbegin() == const_iterator{vec.data()});

    // end points to the element at data() + 42
    CHECK(vec.end() == iterator{vec.data() + 42});
    CHECK(cuda::std::as_const(vec).end() == const_iterator{vec.data() + 42});
    CHECK(vec.cend() == const_iterator{vec.data() + 42});

    // begin and end are not equal
    CHECK(vec.begin() != vec.end());
    CHECK(cuda::std::as_const(vec).begin() != cuda::std::as_const(vec).end());
    CHECK(vec.cbegin() != vec.cend());
  }

  SECTION("cudax::async_mdarray::rbegin/rend properties")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Array&>().rbegin()), reverse_iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Array&>().rbegin()), const_reverse_iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Array&>().crbegin()), const_reverse_iterator>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Array&>().rbegin()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Array&>().rbegin()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<Array&>().crbegin()));

    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Array&>().rend()), reverse_iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Array&>().rend()), const_reverse_iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Array&>().crend()), const_reverse_iterator>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Array&>().rend()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Array&>().rend()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<Array&>().crend()));
  }

  SECTION("cudax::async_mdarray::rbegin/rend no allocation")
  {
    Array vec{env, 0};
    CHECK(vec.rbegin() == reverse_iterator{iterator{nullptr}});
    CHECK(cuda::std::as_const(vec).rbegin() == const_reverse_iterator{const_iterator{nullptr}});
    CHECK(vec.crbegin() == const_reverse_iterator{const_iterator{nullptr}});

    CHECK(vec.rend() == reverse_iterator{iterator{nullptr}});
    CHECK(cuda::std::as_const(vec).rend() == const_reverse_iterator{const_iterator{nullptr}});
    CHECK(vec.crend() == const_reverse_iterator{const_iterator{nullptr}});

    CHECK(vec.rbegin() == vec.rend());
    CHECK(cuda::std::as_const(vec).rbegin() == cuda::std::as_const(vec).rend());
    CHECK(vec.crbegin() == vec.crend());
  }

  SECTION("cudax::async_mdarray::rbegin/rend with allocation")
  {
    Array vec{env, 42, cudax::uninit}; // Note we do not care about the elements just the sizes
    // rbegin points to the element at data() + 42
    CHECK(vec.rbegin() == reverse_iterator{iterator{vec.data() + 42}});
    CHECK(cuda::std::as_const(vec).rbegin() == const_reverse_iterator{const_iterator{vec.data() + 42}});
    CHECK(vec.crbegin() == const_reverse_iterator{const_iterator{vec.data() + 42}});

    // rend points to the element at data()
    CHECK(vec.rend() == reverse_iterator{iterator{vec.data()}});
    CHECK(cuda::std::as_const(vec).rend() == const_reverse_iterator{const_iterator{vec.data()}});
    CHECK(vec.crend() == const_reverse_iterator{const_iterator{vec.data()}});

    // begin and end are not equal
    CHECK(vec.rbegin() != vec.rend());
    CHECK(cuda::std::as_const(vec).rbegin() != cuda::std::as_const(vec).rend());
    CHECK(vec.crbegin() != vec.crend());
  }
}
