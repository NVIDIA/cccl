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

TEMPLATE_TEST_CASE("cudax::vector iterators",
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

  using iterator       = typename extract_properties<TestType>::iterator;
  using const_iterator = typename extract_properties<TestType>::const_iterator;

  using reverse_iterator       = cuda::std::reverse_iterator<iterator>;
  using const_reverse_iterator = cuda::std::reverse_iterator<const_iterator>;

  Resource raw_resource{};
  Resource_ref resource{raw_resource};

  SECTION("cudax::vector::begin/end properties")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Vector&>().begin()), iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Vector&>().begin()), const_iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Vector&>().cbegin()), const_iterator>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Vector&>().begin()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Vector&>().begin()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<Vector&>().cbegin()));

    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Vector&>().end()), iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Vector&>().end()), const_iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Vector&>().cend()), const_iterator>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Vector&>().end()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Vector&>().end()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<Vector&>().cend()));
  }

  SECTION("cudax::vector::begin/end no allocation")
  {
    Vector vec{resource, 0};
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

  SECTION("cudax::vector::begin/end with allocation")
  {
    Vector vec{resource, 42, cudax::uninit}; // Note we do not care about the elements just the sizes
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

  SECTION("cudax::vector::begin/end with allocation after clear")
  {
    Vector vec{resource, 42, cudax::uninit}; // Note we do not care about the elements just the sizes
    auto* ptr = vec.data();
    vec.clear();
    CHECK(vec.data() == ptr);

    // begin points to the element at data
    CHECK(vec.begin() == iterator{ptr});
    CHECK(cuda::std::as_const(vec).begin() == const_iterator{ptr});
    CHECK(vec.cbegin() == const_iterator{ptr});

    // end points to the element at data()
    CHECK(vec.end() == iterator{vec.data()});
    CHECK(cuda::std::as_const(vec).end() == const_iterator{vec.data()});
    CHECK(vec.cend() == const_iterator{vec.data()});

    // begin and end are now equal
    CHECK(vec.begin() == vec.end());
    CHECK(cuda::std::as_const(vec).begin() == cuda::std::as_const(vec).end());
    CHECK(vec.cbegin() == vec.cend());
  }

  SECTION("cudax::vector::rbegin/rend properties")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Vector&>().rbegin()), reverse_iterator>);
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<const Vector&>().rbegin()), const_reverse_iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Vector&>().crbegin()), const_reverse_iterator>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Vector&>().rbegin()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Vector&>().rbegin()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<Vector&>().crbegin()));

    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Vector&>().rend()), reverse_iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Vector&>().rend()), const_reverse_iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Vector&>().crend()), const_reverse_iterator>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Vector&>().rend()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Vector&>().rend()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<Vector&>().crend()));
  }

  SECTION("cudax::vector::rbegin/rend no allocation")
  {
    Vector vec{resource, 0};
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

  SECTION("cudax::vector::rbegin/rend with allocation")
  {
    Vector vec{resource, 42, cudax::uninit}; // Note we do not care about the elements just the sizes
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

  SECTION("cudax::vector::begin/end with allocation after clear")
  {
    Vector vec{resource, 42, cudax::uninit}; // Note we do not care about the elements just the sizes
    auto* ptr = vec.data();
    vec.clear();
    CHECK(vec.data() == ptr);

    // rbegin points to the element at data
    CHECK(vec.rbegin() == reverse_iterator{iterator{ptr}});
    CHECK(cuda::std::as_const(vec).rbegin() == const_reverse_iterator{const_iterator{ptr}});
    CHECK(vec.crbegin() == const_reverse_iterator{const_iterator{ptr}});

    // rend points to the element at data()
    CHECK(vec.rend() == reverse_iterator{iterator{ptr}});
    CHECK(cuda::std::as_const(vec).rend() == const_reverse_iterator{const_iterator{ptr}});
    CHECK(vec.crend() == const_reverse_iterator{const_iterator{ptr}});

    // begin and end are now equal
    CHECK(vec.rbegin() == vec.rend());
    CHECK(cuda::std::as_const(vec).rbegin() == cuda::std::as_const(vec).rend());
    CHECK(vec.crbegin() == vec.crend());
  }
}
