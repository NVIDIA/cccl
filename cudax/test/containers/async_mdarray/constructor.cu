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

#include <cuda/experimental/container.cuh>

#include <stdexcept>

#include "helper.h"
#include "types.h"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE("cudax::async_mdarray constructors",
                   "[container][async_mdarray]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Env           = typename extract_properties<TestType>::env;
  using Resource      = typename extract_properties<TestType>::resource;
  using async_mdarray = typename extract_properties<TestType>::async_mdarray;
  using T             = typename async_mdarray::value_type;

  cudax::stream stream{};
  Env env{Resource{}, stream};

  SECTION("Construction with explicit size")
  {
    { // from env, no alllocation
      const async_mdarray vec{env};
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // from env and size, no alllocation
      const async_mdarray vec{env, 0};
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // from env, size and value, no alllocation
      const async_mdarray vec{env, 0, T{42}};
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // from env and size
      const async_mdarray vec{env, 5};
      CHECK(vec.capacity() == 5);
      CHECK(equal_size_value(vec, 5, T(0)));
    }

    { // from env, size and value
      const async_mdarray vec{env, 5, T{42}};
      CHECK(vec.capacity() == 5);
      CHECK(equal_size_value(vec, 5, T(42)));
    }
  }

  SECTION("Construction from iterators")
  {
    const cuda::std::array<T, 6> input{T(1), T(42), T(1337), T(0), T(12), T(-1)};
    { // can be constructed from two equal forward iterators
      using iter = forward_iterator<const T*>;
      async_mdarray vec(env, iter{input.begin()}, iter{input.begin()});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // can be constructed from two forward iterators
      using iter = forward_iterator<const T*>;
      async_mdarray vec(env, iter{input.begin()}, iter{input.end()});
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
    }

    { // can be constructed from two input iterators
      async_mdarray vec(env, input.begin(), input.end());
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
    }
  }

  SECTION("Construction from range")
  {
    { // can be constructed from an empty uncommon forward range
      async_mdarray vec(env, uncommon_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // can be constructed from a non-empty uncommon forward range
      async_mdarray vec(env, uncommon_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // can be constructed from an empty sized uncommon forward range
      async_mdarray vec(env, sized_uncommon_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // can be constructed from a non-empty sized uncommon forward range
      async_mdarray vec(env, sized_uncommon_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // can be constructed from an empty random access range
      async_mdarray vec(env, cuda::std::array<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // can be constructed from a non-empty random access range
      async_mdarray vec(env, cuda::std::array<T, 6>{T(1), T(42), T(1337), T(0), T(12), T(-1)});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }
  }

  SECTION("Construction from initializer_list")
  {
    { // can be constructed from an empty initializer_list
      const cuda::std::initializer_list<T> input{};
      async_mdarray vec(env, input);
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // can be constructed from a non-empty initializer_list
      const cuda::std::initializer_list<T> input{T(1), T(42), T(1337), T(0), T(12), T(-1)};
      async_mdarray vec(env, input);
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec));
    }
  }

  SECTION("copy construction")
  {
    static_assert(!cuda::std::is_nothrow_copy_constructible<async_mdarray>::value, "");
    { // can be copy constructed from empty input
      const async_mdarray input{env, 0};
      async_mdarray vec(input);
      CHECK(vec.empty());
    }

    { // can be copy constructed from non-empty input
      const async_mdarray input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      async_mdarray vec(input);
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }
  }

  SECTION("move construction")
  {
    static_assert(cuda::std::is_nothrow_move_constructible<async_mdarray>::value, "");

    { // can be move constructed with empty input
      async_mdarray input{env, 0};
      async_mdarray vec(cuda::std::move(input));
      CHECK(vec.empty());
      CHECK(input.empty());
    }

    { // can be move constructed from non-empty input
      async_mdarray input{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};

      // ensure that we steal the data
      const auto* allocation = input.data();
      async_mdarray vec(cuda::std::move(input));
      CHECK(vec.capacity() == 6);
      CHECK(vec.data() == allocation);
      CHECK(input.capacity() == 0);
      CHECK(input.data() == nullptr);
      CHECK(equal_range(vec));
    }
  }

#if 0 // Implement exception handling
#  ifndef TEST_HAS_NO_EXCEPTIONS
  SECTION("Exception handling throwing bad_alloc")
  {
    using async_mdarray = cudax::async_mdarray<int>;

    try
    {
      async_mdarray too_small(2 * capacity);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }

    try
    {
      async_mdarray too_small(2 * capacity, 42);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }

    try
    {
      cuda::std::array<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
      async_mdarray too_small(input.begin(), input.end());
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }

    try
    {
      cuda::std::initializer_list<int> input{0, 1, 2, 3, 4, 5, 6};
      async_mdarray too_small(input);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }

    try
    {
      uncommon_range<int, 2 * capacity> input{{0, 1, 2, 3, 4, 5, 6, 7}};
      async_mdarray too_small(input);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }

    try
    {
      sized_uncommon_range<int, 2 * capacity> input{{0, 1, 2, 3, 4, 5, 6, 7}};
      async_mdarray too_small(input);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }

    try
    {
      cuda::std::array<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
      async_mdarray too_small(input);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }
  }
#  endif // !TEST_HAS_NO_EXCEPTIONS
#endif // 0
}
