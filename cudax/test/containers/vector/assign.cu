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

#include <cuda/experimental/vector.cuh>

#include <stdexcept>

#include "helper.h"
#include "types.h"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE("cudax::vector assign",
                   "[container][vector]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
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
      Vector vec{resource, 10, T(-2)};
      vec.assign_range(input_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // cudax::vector::assign_range with a non-empty input, shrinking
      Vector vec{resource, 10, T(-2)};
      vec.assign_range(input_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // cudax::vector::assign_range with a non-empty input, growing
      Vector vec{resource, 4, T(-2)};
      vec.assign_range(input_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // cudax::vector::assign_range with a non-empty input, growing, no reallocation
      Vector vec{resource, 10, T(-2)};
      vec.resize(2);
      vec.assign_range(input_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
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
      Vector vec{resource, 10, T(-2)};
      vec.assign_range(uncommon_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // cudax::vector::assign_range with a non-empty input, shrinking
      Vector vec{resource, 10, T(-2)};
      vec.assign_range(uncommon_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // cudax::vector::assign_range with a non-empty input, growing
      Vector vec{resource, 4, T(-2)};
      vec.assign_range(uncommon_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // cudax::vector::assign_range with a non-empty input, growing, no reallocation
      Vector vec{resource, 10, T(-2)};
      vec.resize(2);
      vec.assign_range(uncommon_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
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
      Vector vec{resource, 10, T(-2)};
      vec.assign_range(sized_uncommon_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // cudax::vector::assign_range with a non-empty input, shrinking
      Vector vec{resource, 10, T(-2)};
      vec.assign_range(sized_uncommon_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // cudax::vector::assign_range with a non-empty input, growing
      Vector vec{resource, 4, T(-2)};
      vec.assign_range(sized_uncommon_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // cudax::vector::assign_range with a non-empty input, growing, no reallocation
      Vector vec{resource, 10, T(-2)};
      vec.resize(2);
      vec.assign_range(sized_uncommon_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
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
      Vector vec{resource, 10, T(-2)};
      vec.assign_range(cuda::std::array<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // cudax::vector::assign_range with a non-empty input, shrinking
      Vector vec{resource, 10, T(-2)};
      vec.assign_range(cuda::std::array<T, 6>{T(1), T(42), T(1337), T(0), T(12), T(-1)});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // cudax::vector::assign_range with a non-empty input, growing
      Vector vec{resource, 4, T(-2)};
      vec.assign_range(cuda::std::array<T, 6>{T(1), T(42), T(1337), T(0), T(12), T(-1)});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // cudax::vector::assign_range with a non-empty input, growing, no reallocation
      Vector vec{resource, 10, T(-2)};
      vec.resize(2);
      vec.assign_range(cuda::std::array<T, 6>{T(1), T(42), T(1337), T(0), T(12), T(-1)});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
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
      Vector vec{resource, 10, T(-2)};
      vec.assign(0, T(42));
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // cudax::vector::assign(count, const T&), shrinking
      Vector vec{resource, 10, T(-2)};
      vec.assign(2, T(42));
      CHECK(!vec.empty());
      CHECK(equal_size_value(vec, 2, T(42)));
    }

    { // cudax::vector::assign(count, const T&), growing
      Vector vec{resource, 4, T(-2)};
      vec.assign(6, T(42));
      CHECK(!vec.empty());
      CHECK(equal_size_value(vec, 6, T{42}));
    }
  }

  SECTION("cudax::vector::assign(iter, iter) input iterators")
  {
    const cuda::std::array<T, 6> input = {T(1), T(42), T(1337), T(0), T(12), T(-1)};
    using iter                         = cpp17_input_iterator<const T*>;
    { // cudax::vector::assign(iter, iter), with input iterators empty range
      Vector vec{resource};
      vec.assign(iter{input.begin()}, iter{input.begin()});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::assign(iter, iter), with input iterators shrink to empty range
      Vector vec{resource, 4, T(-2)};
      vec.assign(iter{input.begin()}, iter{input.begin()});
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // cudax::vector::assign(iter, iter), with input iterators shrinking
      Vector vec{resource, 10, T(-2)};
      vec.assign(iter{input.begin()}, iter{input.end()});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // cudax::vector::assign(iter, iter), with input iterators growing
      Vector vec{resource, 4, T(-2)};
      vec.assign(iter{input.begin()}, iter{input.end()});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }
  }

  SECTION("cudax::vector::assign(iter, iter) forward iterators")
  {
    const cuda::std::array<T, 6> input = {T(1), T(42), T(1337), T(0), T(12), T(-1)};
    { // cudax::vector::assign(iter, iter), with forward iterators empty range
      Vector vec{resource};
      vec.assign(input.begin(), input.begin());
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::assign(iter, iter), with forward iterators shrinking to empty
      Vector vec{resource, 10, T(-2)};
      vec.assign(input.begin(), input.begin());
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // cudax::vector::assign(iter, iter), with forward iterators shrinking
      Vector vec{resource, 10, T(-2)};
      vec.assign(input.begin(), input.end());
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // cudax::vector::assign(iter, iter), with forward iterators growing
      Vector vec{resource, 4, T(-2)};
      vec.assign(input.begin(), input.end());
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }
  }

  SECTION("cudax::vector::assign(initializer_list)")
  {
    { // cudax::vector::assign(initializer_list), empty range
      Vector vec{resource};
      vec.assign(cuda::std::initializer_list<T>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::assign(initializer_list), shrinking to empty
      Vector vec{resource, 10, T(-2)};
      vec.assign(cuda::std::initializer_list<T>{});
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // cudax::vector::assign(initializer_list), shrinking
      Vector vec{resource, 10, T(-2)};
      vec.assign(cuda::std::initializer_list<T>{T(1), T(42), T(1337), T(0), T(12), T(-1)});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
    }

    { // cudax::vector::assign(initializer_list), growing
      Vector vec{resource, 4, T(-2)};
      vec.assign(cuda::std::initializer_list<T>{T(1), T(42), T(1337), T(0), T(12), T(-1)});
      CHECK(!vec.empty());
      CHECK(equal_range(vec));
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
