//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

TEMPLATE_TEST_CASE("cudax::async_buffer iterators",
                   "[container][async_buffer]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Env       = typename extract_properties<TestType>::env;
  using Resource  = typename extract_properties<TestType>::resource;
  using Buffer    = typename extract_properties<TestType>::async_buffer;
  using T         = typename Buffer::value_type;
  using size_type = typename Buffer::size_type;

  using iterator       = typename extract_properties<TestType>::iterator;
  using const_iterator = typename extract_properties<TestType>::const_iterator;

  using reverse_iterator       = cuda::std::reverse_iterator<iterator>;
  using const_reverse_iterator = cuda::std::reverse_iterator<const_iterator>;

  cudax::stream stream{};
  Env env{Resource{}, stream};

  SECTION("cudax::async_buffer::begin/end properties")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().begin()), iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Buffer&>().begin()), const_iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().cbegin()), const_iterator>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Buffer&>().begin()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Buffer&>().begin()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<Buffer&>().cbegin()));

    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().end()), iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Buffer&>().end()), const_iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().cend()), const_iterator>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Buffer&>().end()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Buffer&>().end()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<Buffer&>().cend()));
  }

  SECTION("cudax::async_buffer::begin/end no allocation")
  {
    Buffer buf{env, 0};
    CHECK(buf.begin() == iterator{nullptr});
    CHECK(cuda::std::as_const(buf).begin() == const_iterator{nullptr});
    CHECK(buf.cbegin() == const_iterator{nullptr});

    CHECK(buf.end() == iterator{nullptr});
    CHECK(cuda::std::as_const(buf).end() == const_iterator{nullptr});
    CHECK(buf.cend() == const_iterator{nullptr});

    CHECK(buf.begin() == buf.end());
    CHECK(cuda::std::as_const(buf).begin() == cuda::std::as_const(buf).end());
    CHECK(buf.cbegin() == buf.cend());
  }

  SECTION("cudax::async_buffer::begin/end with allocation")
  {
    Buffer buf{env, 42, cudax::uninit}; // Note we do not care about the elements just the sizes
    // begin points to the element at data()
    CHECK(buf.begin() == iterator{buf.data()});
    CHECK(cuda::std::as_const(buf).begin() == const_iterator{buf.data()});
    CHECK(buf.cbegin() == const_iterator{buf.data()});

    // end points to the element at data() + 42
    CHECK(buf.end() == iterator{buf.data() + 42});
    CHECK(cuda::std::as_const(buf).end() == const_iterator{buf.data() + 42});
    CHECK(buf.cend() == const_iterator{buf.data() + 42});

    // begin and end are not equal
    CHECK(buf.begin() != buf.end());
    CHECK(cuda::std::as_const(buf).begin() != cuda::std::as_const(buf).end());
    CHECK(buf.cbegin() != buf.cend());
  }

  SECTION("cudax::async_buffer::rbegin/rend properties")
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().rbegin()), reverse_iterator>);
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<const Buffer&>().rbegin()), const_reverse_iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().crbegin()), const_reverse_iterator>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Buffer&>().rbegin()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Buffer&>().rbegin()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<Buffer&>().crbegin()));

    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().rend()), reverse_iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const Buffer&>().rend()), const_reverse_iterator>);
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().crend()), const_reverse_iterator>);
    STATIC_REQUIRE(noexcept(cuda::std::declval<Buffer&>().rend()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<const Buffer&>().rend()));
    STATIC_REQUIRE(noexcept(cuda::std::declval<Buffer&>().crend()));
  }

  SECTION("cudax::async_buffer::rbegin/rend no allocation")
  {
    Buffer buf{env, 0};
    CHECK(buf.rbegin() == reverse_iterator{iterator{nullptr}});
    CHECK(cuda::std::as_const(buf).rbegin() == const_reverse_iterator{const_iterator{nullptr}});
    CHECK(buf.crbegin() == const_reverse_iterator{const_iterator{nullptr}});

    CHECK(buf.rend() == reverse_iterator{iterator{nullptr}});
    CHECK(cuda::std::as_const(buf).rend() == const_reverse_iterator{const_iterator{nullptr}});
    CHECK(buf.crend() == const_reverse_iterator{const_iterator{nullptr}});

    CHECK(buf.rbegin() == buf.rend());
    CHECK(cuda::std::as_const(buf).rbegin() == cuda::std::as_const(buf).rend());
    CHECK(buf.crbegin() == buf.crend());
  }

  SECTION("cudax::async_buffer::rbegin/rend with allocation")
  {
    Buffer buf{env, 42, cudax::uninit}; // Note we do not care about the elements just the sizes
    // rbegin points to the element at data() + 42
    CHECK(buf.rbegin() == reverse_iterator{iterator{buf.data() + 42}});
    CHECK(cuda::std::as_const(buf).rbegin() == const_reverse_iterator{const_iterator{buf.data() + 42}});
    CHECK(buf.crbegin() == const_reverse_iterator{const_iterator{buf.data() + 42}});

    // rend points to the element at data()
    CHECK(buf.rend() == reverse_iterator{iterator{buf.data()}});
    CHECK(cuda::std::as_const(buf).rend() == const_reverse_iterator{const_iterator{buf.data()}});
    CHECK(buf.crend() == const_reverse_iterator{const_iterator{buf.data()}});

    // begin and end are not equal
    CHECK(buf.rbegin() != buf.rend());
    CHECK(cuda::std::as_const(buf).rbegin() != cuda::std::as_const(buf).rend());
    CHECK(buf.crbegin() != buf.crend());
  }
}
