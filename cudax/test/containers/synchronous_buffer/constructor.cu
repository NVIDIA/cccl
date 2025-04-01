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

#include <cuda/experimental/container.cuh>

#include <stdexcept>

#include "helper.h"
#include "types.h"

TEMPLATE_TEST_CASE("cudax::synchronous_buffer constructors",
                   "[container][synchronous_buffer]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Resource = typename extract_properties<TestType>::resource;
  using Buffer   = typename extract_properties<TestType>::synchronous_buffer;
  using T        = typename Buffer::value_type;

  Resource resource{};

  SECTION("Construction with explicit size")
  {
    { // from resource, no allocation
      const Buffer buf{resource};
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // from resource and size, no allocation
      const Buffer buf{resource, 0};
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // from resource, size and value, no allocation
      const Buffer buf{resource, 0, T{42}};
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // from resource and size
      const Buffer buf{resource, 5};
      CUDAX_CHECK(buf.size() == 5);
      CUDAX_CHECK(equal_size_value(buf, 5, T(0)));
    }

    { // from resource, size and value
      const Buffer buf{resource, 5, T{42}};
      CUDAX_CHECK(buf.size() == 5);
      CUDAX_CHECK(equal_size_value(buf, 5, T(42)));
    }
  }

  SECTION("Construction from iterators")
  {
    const cuda::std::array<T, 6> input{T(1), T(42), T(1337), T(0), T(12), T(-1)};
    { // can be constructed from two equal forward iterators
      using iter = forward_iterator<const T*>;
      Buffer buf(resource, iter{input.begin()}, iter{input.begin()});
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // can be constructed from two forward iterators
      using iter = forward_iterator<const T*>;
      Buffer buf(resource, iter{input.begin()}, iter{input.end()});
      CUDAX_CHECK(buf.size() == 6);
      CUDAX_CHECK(equal_range(buf));
    }

    { // can be constructed from two input iterators
      Buffer buf(resource, input.begin(), input.end());
      CUDAX_CHECK(buf.size() == 6);
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("Construction from range")
  {
    { // can be constructed from an empty uncommon forward range
      Buffer buf(resource, uncommon_range<T, 0>{});
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // can be constructed from a non-empty uncommon forward range
      Buffer buf(resource, uncommon_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }

    { // can be constructed from an empty sized uncommon forward range
      Buffer buf(resource, sized_uncommon_range<T, 0>{});
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // can be constructed from a non-empty sized uncommon forward range
      Buffer buf(resource, sized_uncommon_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }

    { // can be constructed from an empty random access range
      Buffer buf(resource, cuda::std::array<T, 0>{});
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // can be constructed from a non-empty random access range
      Buffer buf(resource, cuda::std::array<T, 6>{T(1), T(42), T(1337), T(0), T(12), T(-1)});
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("Construction from initializer_list")
  {
    { // can be constructed from an empty initializer_list
      const cuda::std::initializer_list<T> input{};
      Buffer buf(resource, input);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // can be constructed from a non-empty initializer_list
      const cuda::std::initializer_list<T> input{T(1), T(42), T(1337), T(0), T(12), T(-1)};
      Buffer buf(resource, input);
      CUDAX_CHECK(buf.size() == 6);
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("copy construction")
  {
    static_assert(!cuda::std::is_nothrow_copy_constructible<Buffer>::value, "");
    { // can be copy constructed from empty input
      const Buffer input{resource, 0};
      Buffer buf(input);
      CUDAX_CHECK(buf.empty());
    }

    { // can be copy constructed from non-empty input
      const Buffer input{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      Buffer buf(input);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("move construction")
  {
    static_assert(cuda::std::is_nothrow_move_constructible<Buffer>::value, "");

    { // can be move constructed with empty input
      Buffer input{resource, 0};
      Buffer buf(cuda::std::move(input));
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(input.empty());
    }

    { // can be move constructed from non-empty input
      Buffer input{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};

      // ensure that we steal the data
      const auto* allocation = input.data();
      Buffer buf(cuda::std::move(input));
      CUDAX_CHECK(buf.size() == 6);
      CUDAX_CHECK(buf.data() == allocation);
      CUDAX_CHECK(input.size() == 0);
      CUDAX_CHECK(input.data() == nullptr);
      CUDAX_CHECK(equal_range(buf));
    }
  }

#if 0 // Implement exception handling
#  ifndef TEST_HAS_NO_EXCEPTIONS
  SECTION("Exception handling throwing bad_alloc")
  {
    using synchronous_buffer = cudax::synchronous_buffer<int>;

    try
    {
      synchronous_buffer too_small(2 * capacity);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CUDAX_CHECK(false);
    }

    try
    {
      synchronous_buffer too_small(2 * capacity, 42);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CUDAX_CHECK(false);
    }

    try
    {
      cuda::std::array<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
      synchronous_buffer too_small(input.begin(), input.end());
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CUDAX_CHECK(false);
    }

    try
    {
      cuda::std::initializer_list<int> input{0, 1, 2, 3, 4, 5, 6};
      synchronous_buffer too_small(input);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CUDAX_CHECK(false);
    }

    try
    {
      uncommon_range<int, 2 * capacity> input{{0, 1, 2, 3, 4, 5, 6, 7}};
      synchronous_buffer too_small(input);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CUDAX_CHECK(false);
    }

    try
    {
      sized_uncommon_range<int, 2 * capacity> input{{0, 1, 2, 3, 4, 5, 6, 7}};
      synchronous_buffer too_small(input);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CUDAX_CHECK(false);
    }

    try
    {
      cuda::std::array<int, 2 * capacity> input{0, 1, 2, 3, 4, 5, 6, 7};
      synchronous_buffer too_small(input);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CUDAX_CHECK(false);
    }
  }
#  endif // !TEST_HAS_NO_EXCEPTIONS
#endif // 0
}
