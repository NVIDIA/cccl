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

#include <cuda/experimental/vector>

#include <stdexcept>

#include "types.h"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE(
  "cudax::vector assign", "[container][vector]", cuda::std::tuple<>, cuda::std::tuple<cuda::mr::host_accessible>)
{
  using Resource     = typename extract_properties<TestType>::resource;
  using Resource_ref = typename extract_properties<TestType>::resource_ref;
  using Vector       = typename extract_properties<TestType>::vector;
  using T            = typename Vector::value_type;

  Resource raw_resource{};
  Resource_ref resource{raw_resource};

  SECTION("cudax::vector::assign_range input range")
  {
    { // cudax::vector::assign_range with an empty input
      Vector vec{resource};
      vec.assign_range(input_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::assign_range with an empty input, shrinking
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign_range(input_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // cudax::vector::assign_range with a non-empty input, shrinking
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign_range(input_range<T, 2>{{T(42), T(42)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec, cuda::std::array<T, 2>{T(42), T(42)}));
    }

    { // cudax::vector::assign_range with a non-empty input, growing
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign_range(input_range<T, 6>{{T(42), T(1), T(42), T(1337), T(0), T(42)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec, cuda::std::array<T, 6>{T(42), T(1), T(42), T(1337), T(0), T(42)}));
    }

    { // cudax::vector::assign_range with a non-empty input, growing, no reallocation
      Vector vec{resource, 42, T{5}};
      vec.resize(2);
      vec.assign_range(input_range<T, 6>{{T(42), T(1), T(42), T(1337), T(0), T(42)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec, cuda::std::array<T, 6>{T(42), T(1), T(42), T(1337), T(0), T(42)}));
    }
  }

  SECTION("cudax::vector::assign_range uncommon range")
  {
    { // cudax::vector::assign_range with an empty input
      Vector vec{resource};
      vec.assign_range(uncommon_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::assign_range with an empty input, shrinking
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign_range(uncommon_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // cudax::vector::assign_range with a non-empty input, shrinking
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign_range(uncommon_range<T, 2>{{T(42), T(42)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec, cuda::std::array<T, 2>{T(42), T(42)}));
    }

    { // cudax::vector::assign_range with a non-empty input, growing
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign_range(uncommon_range<T, 6>{{T(42), T(1), T(42), T(1337), T(0), T(42)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec, cuda::std::array<T, 6>{T(42), T(1), T(42), T(1337), T(0), T(42)}));
    }

    { // cudax::vector::assign_range with a non-empty input, growing, no reallocation
      Vector vec{resource, 42, T{5}};
      vec.resize(2);
      vec.assign_range(uncommon_range<T, 6>{{T(42), T(1), T(42), T(1337), T(0), T(42)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec, cuda::std::array<T, 6>{T(42), T(1), T(42), T(1337), T(0), T(42)}));
    }
  }

  SECTION("cudax::vector::assign_range sized uncommon range")
  {
    { // cudax::vector::assign_range with an empty input
      Vector vec{resource};
      vec.assign_range(sized_uncommon_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::assign_range with an empty input, shrinking
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign_range(sized_uncommon_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // cudax::vector::assign_range with a non-empty input, shrinking
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign_range(sized_uncommon_range<T, 2>{{T(42), T(42)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec, cuda::std::array<T, 2>{T(42), T(42)}));
    }

    { // cudax::vector::assign_range with a non-empty input, growing
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign_range(sized_uncommon_range<T, 6>{{T(42), T(1), T(42), T(1337), T(0), T(42)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec, cuda::std::array<T, 6>{T(42), T(1), T(42), T(1337), T(0), T(42)}));
    }

    { // cudax::vector::assign_range with a non-empty input, growing, no reallocation
      Vector vec{resource, 42, T{5}};
      vec.resize(2);
      vec.assign_range(sized_uncommon_range<T, 6>{{T(42), T(1), T(42), T(1337), T(0), T(42)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec, cuda::std::array<T, 6>{T(42), T(1), T(42), T(1337), T(0), T(42)}));
    }
  }

  SECTION("cudax::vector::assign_range random access range")
  {
    { // cudax::vector::assign_range with an empty input
      Vector vec{resource};
      vec.assign_range(cuda::std::array<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::assign_range with an empty input, shrinking
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign_range(cuda::std::array<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // cudax::vector::assign_range with a non-empty input, shrinking
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign_range(cuda::std::array<T, 2>{T(42), T(42)});
      CHECK(!vec.empty());
      CHECK(equal_range(vec, cuda::std::array<T, 2>{T(42), T(42)}));
    }

    { // cudax::vector::assign_range with a non-empty input, growing
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign_range(cuda::std::array<T, 6>{T(42), T(1), T(42), T(1337), T(0), T(42)});
      CHECK(!vec.empty());
      CHECK(equal_range(vec, cuda::std::array<T, 6>{T(42), T(1), T(42), T(1337), T(0), T(42)}));
    }

    { // cudax::vector::assign_range with a non-empty input, growing, no reallocation
      Vector vec{resource, 42, T{5}};
      vec.resize(2);
      vec.assign_range(cuda::std::array<T, 6>{T(42), T(1), T(42), T(1337), T(0), T(42)});
      CHECK(!vec.empty());
      CHECK(equal_range(vec, cuda::std::array<T, 6>{T(42), T(1), T(42), T(1337), T(0), T(42)}));
    }
  }

  SECTION("cudax::vector::assign(count, const T&)")
  {
    { // cudax::vector::assign(count, const T&), zero count from empty
      Vector vec{resource};
      vec.assign(0, T(42));
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::assign(count, const T&), shrinking to empty
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign(0, T(42));
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // cudax::vector::assign(count, const T&), shrinking
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign(2, T(42));
      CHECK(!vec.empty());
      CHECK(equal_range(vec, cuda::std::array<T, 2>{T(42), T(42)}));
    }

    { // cudax::vector::assign(count, const T&), growing
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign(6, T(42));
      CHECK(!vec.empty());
      CHECK(equal_range(vec, cuda::std::array<T, 6>{T(42), T(42), T(42), T(42), T(42), T(42)}));
    }
  }

  SECTION("cudax::vector::assign(iter, iter) input iterators")
  {
    using iter = cpp17_input_iterator<const T*>;
    { // cudax::vector::assign(iter, iter), with input iterators empty range
      const cuda::std::array<T, 0> expected = {};
      Vector vec{resource};
      vec.assign(iter{expected.begin()}, iter{expected.end()});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::assign(iter, iter), with input iterators shrink to empty range
      const cuda::std::array<T, 0> expected = {};
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign(iter{expected.begin()}, iter{expected.end()});
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // cudax::vector::assign(iter, iter), with input iterators shrinking
      const cuda::std::array<T, 2> expected = {T(42), T(42)};
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign(iter{expected.begin()}, iter{expected.end()});
      CHECK(!vec.empty());
      CHECK(equal_range(vec, expected));
    }

    { // cudax::vector::assign(iter, iter), with input iterators growing
      const cuda::std::array<T, 6> expected = {T(42), T(1), T(42), T(1337), T(0), T(42)};
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign(iter{expected.begin()}, iter{expected.end()});
      CHECK(!vec.empty());
      CHECK(equal_range(vec, expected));
    }
  }

  SECTION("cudax::vector::assign(iter, iter) forward iterators")
  {
    { // cudax::vector::assign(iter, iter), with forward iterators empty range
      const cuda::std::array<T, 0> expected = {};
      Vector vec{resource};
      vec.assign(expected.begin(), expected.end());
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::assign(iter, iter), with forward iterators shrinking to empty
      const cuda::std::array<T, 0> expected = {};
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign(expected.begin(), expected.end());
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // cudax::vector::assign(iter, iter), with forward iterators shrinking
      const cuda::std::array<T, 2> expected = {T(42), T(42)};
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign(expected.begin(), expected.end());
      CHECK(!vec.empty());
      CHECK(equal_range(vec, expected));
    }

    { // cudax::vector::assign(iter, iter), with forward iterators growing
      const cuda::std::array<T, 6> expected = {T(42), T(1), T(42), T(1337), T(0), T(42)};
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign(expected.begin(), expected.end());
      CHECK(!vec.empty());
      CHECK(equal_range(vec, expected));
    }
  }

  SECTION("cudax::vector::assign(initializer_list)")
  {
    { // cudax::vector::assign(initializer_list), empty range
      const cuda::std::initializer_list<T> expected = {};
      Vector vec{resource};
      vec.assign(expected);
      CHECK(vec.empty());
    }

    { // cudax::vector::assign(initializer_list), shrinking to empty
      const cuda::std::initializer_list<T> expected = {};
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign(expected);
      CHECK(vec.empty());
    }

    { // cudax::vector::assign(initializer_list), shrinking
      const cuda::std::initializer_list<T> expected = {T(42), T(42)};
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign(expected);
      CHECK(!vec.empty());
      CHECK(equal_range(vec, expected));
    }

    { // cudax::vector::assign(initializer_list), growing
      const cuda::std::initializer_list<T> expected = {T(42), T(1), T(42), T(1337), T(0), T(42)};
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.assign(expected);
      CHECK(!vec.empty());
      CHECK(equal_range(vec, expected));
    }
  }

#if 0 // Implement exceptions
#  ifndef TEST_HAS_NO_EXCEPTIONS
  SECTION("cudax::vector::assign exception handling")
  {
    try
    {
      too_small.assign(2 * capacity, 42);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }

    try
    {
      using iter = cpp17_input_iterator<const int*>;
      too_small.assign(iter{input.begin()}, iter{input.end()});
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }

    try
    {
      too_small.assign(input.begin(), input.end());
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }

    try
    {
      too_small.assign(cuda::std::initializer_list<int>{0, 1, 2, 3, 4, 5, 6});
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }
  }
#  endif // TEST_HAS_NO_EXCEPTIONS
#endif // Implement exceptions
}
