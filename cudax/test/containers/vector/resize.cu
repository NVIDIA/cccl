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
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/vector>

#include "types.h"
#include <catch2/catch.hpp>

struct is_one
{
  template <class T>
  __host__ __device__ constexpr bool operator()(const T& val) const noexcept
  {
    return val == T(1);
  }
};

// TODO: only device accessible resource
TEMPLATE_TEST_CASE("cudax::vector resize",
                   "[container][vector]",
                   cuda::std::tuple<>,
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Resource      = typename extract_properties<TestType>::resource;
  using Resource_ref  = typename extract_properties<TestType>::resource_ref;
  using OtherResource = typename extract_properties<TestType>::other_resource;
  using Vector        = typename extract_properties<TestType>::vector;
  using T             = typename Vector::value_type;
  using size_type     = typename Vector::size_type;

  using iterator       = typename extract_properties<TestType>::iterator;
  using const_iterator = typename extract_properties<TestType>::const_iterator;

  Resource raw_resource{};
  Resource_ref resource{raw_resource};

  SECTION("cudax::vector::clear")
  {
    { // cudax::vector::clear, from empty
      Vector vec{resource};
      vec.clear();
      CHECK(vec.empty());
    }

    { // cudax::vector::clear, non-empty
      Vector vec{resource, {T(1), T(1337), T(42), T(12), T(0), T(-1)}};
      vec.clear();
      CHECK(vec.empty());
    }
  }

  SECTION("cudax::vector::erase")
  {
    { // cudax::vector::erase(iter)
      Vector vec{resource, {T(1), T(1337), T(42), T(12), T(0), T(-1)}};
      auto res = vec.erase(vec.begin() + 1);
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(*res == T(42));
      CHECK(equal_range(vec, cuda::std::array<T, 5>{T(1), T(42), T(12), T(0), T(-1)}));
    }

    { // cudax::vector::erase(iter, iter), iterators are equal
      Vector vec{resource, {T(1), T(1337), T(42), T(12), T(0), T(-1)}};
      auto res = vec.erase(vec.begin() + 1, vec.begin() + 1);
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(*res == T(1337));
      CHECK(equal_range(vec, cuda::std::array<T, 6>{T(1), T(1337), T(42), T(12), T(0), T(-1)}));
    }

    { // cudax::vector::erase(iter, iter)
      Vector vec{resource, {T(1), T(1337), T(42), T(12), T(0), T(-1)}};
      auto res = vec.erase(vec.begin() + 1, vec.begin() + 3);
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(*res == T(12));
      CHECK(equal_range(vec, cuda::std::array<T, 4>{T(1), T(12), T(0), T(-1)}));
    }
  }

  SECTION("cuda::std::erase(cudax::vector")
  {
    { // erase(cudax::vector, value), empty
      Vector vec{resource};
      auto res = erase(vec, T(1));
      static_assert(cuda::std::is_same<decltype(res), size_type>::value, "");
      CHECK(res == 0);
      CHECK(vec.empty());
    }

    { // erase(cudax::vector, value), non-empty
      Vector vec{resource, {T(1), T(1337), T(1), T(12), T(0), T(-1)}};
      auto res = erase(vec, T(1));
      static_assert(cuda::std::is_same<decltype(res), size_type>::value, "");
      CHECK(res == 2);
      CHECK(equal_range(vec, cuda::std::array<T, 4>{T(1337), T(12), T(0), T(-1)}));
    }

    { // erase(cudax::vector, value), no match
      Vector vec{resource, {T(1), T(1337), T(1), T(12), T(0), T(-1)}};
      auto res = erase(vec, T(5));
      static_assert(cuda::std::is_same<decltype(res), size_type>::value, "");
      CHECK(res == 0);
      CHECK(equal_range(vec, cuda::std::array<T, 6>{T(1), T(1337), T(1), T(12), T(0), T(-1)}));
    }
  }

  SECTION("cudax::vector::erase_if")
  {
    { // cudax::vector::erase_if, empty
      Vector vec{resource};
      auto res = erase_if(vec, is_one{});
      static_assert(cuda::std::is_same<decltype(res), size_type>::value, "");
      CHECK(res == 0);
      CHECK(vec.empty());
    }

    { // cudax::vector::erase_if, non-empty
      Vector vec{resource, {T(1), T(1337), T(1), T(12), T(0), T(-1)}};
      auto res = erase_if(vec, is_one{});
      static_assert(cuda::std::is_same<decltype(res), size_type>::value, "");
      CHECK(res == 2);
      CHECK(equal_range(vec, cuda::std::array<T, 4>{T(1337), T(12), T(0), T(-1)}));
    }

    { // cudax::vector::erase_if, non-empty, no match
      Vector vec{resource, {T(2), T(1337), T(2), T(12), T(0), T(-1)}};
      auto res = erase_if(vec, is_one{});
      static_assert(cuda::std::is_same<decltype(res), size_type>::value, "");
      CHECK(res == 0);
      CHECK(equal_range(vec, cuda::std::array<T, 6>{T(2), T(1337), T(2), T(12), T(0), T(-1)}));
    }
  }

  SECTION("cudax::vector::pop_back")
  {
    { // cudax::vector::pop_back
      Vector vec{resource, {T(1), T(1337), T(42), T(12), T(0), T(-1)}};
      vec.pop_back();
      CHECK(equal_range(vec, cuda::std::array<T, 5>{T(1), T(1337), T(42), T(12), T(0)}));
    }
  }

  SECTION("cudax::vector::resize")
  {
    { // cudax::vector::resize with a size, shrinking
      Vector vec{resource, {T(1), T(1337), T(42), T(12), T(0), T(-1)}};
      vec.resize(1);
      CHECK(equal_range(vec, cuda::std::array<T, 1>{T(1)}));
    }

    { // cudax::vector::resize with a size and value, shrinking
      Vector vec{resource, {T(1), T(1337), T(42), T(12), T(0), T(-1)}};
      vec.resize(1, T(5));
      CHECK(equal_range(vec, cuda::std::array<T, 1>{T(1)}));
    }

    { // cudax::vector::resize with a size and uninit, shrinking
      Vector vec{resource, {T(1), T(1337), T(42), T(12), T(0), T(-1)}};
      vec.resize(1, cudax::uninit);
      CHECK(equal_range(vec, cuda::std::array<T, 1>{T(1)}));
    }

    { // cudax::vector::resize with a size, growing, new elements are value initialized
      Vector vec{resource, {T(1), T(1337), T(42), T(12), T(0), T(-1)}};
      CHECK(vec.capacity() == 6);
      vec.resize(8);
      CHECK(vec.capacity() == 8);
      CHECK(vec.size() == 8);
      CHECK(equal_range(vec, cuda::std::array<T, 8>{T(1), T(1337), T(42), T(12), T(0), T(-1), T(0), T(0)}));
    }

    { // cudax::vector::resize with a size and value, growing, new elements are copied
      Vector vec{resource, {T(1), T(1337), T(42), T(12), T(0), T(-1)}};
      CHECK(vec.capacity() == 6);
      vec.resize(8, T(5));
      CHECK(vec.capacity() == 8);
      CHECK(vec.size() == 8);
      CHECK(equal_range(vec, cuda::std::array<T, 8>{T(1), T(1337), T(42), T(12), T(0), T(-1), T(5), T(5)}));
    }

    { // cudax::vector::resize with a size and uninit, growing, new elements are copied
      Vector vec{resource, {T(1), T(1337), T(42), T(12), T(0), T(-1)}};
      CHECK(vec.capacity() == 6);
      vec.resize(8, cudax::uninit);
      CHECK(vec.capacity() == 8);
      CHECK(vec.size() == 8);

      // The last elements are indeterminate, so we can only check that nothing has changed
      cuda::std::span<T> subspan{vec.data(), 6};
      CHECK(equal_range(subspan, cuda::std::array<T, 6>{T(1), T(1337), T(42), T(12), T(0), T(-1)}));
    }
  }

  SECTION("cudax::vector::shrink_to_fit")
  {
    { // cudax::vector::shrink_to_fit, no allocation
      Vector vec{resource};
      vec.shrink_to_fit();
      CHECK(vec.capacity() == 0);
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::shrink_to_fit, with allocation, but empty
      Vector vec{resource, {T(1), T(1337), T(1), T(12), T(0), T(-1)}};
      vec.clear();
      vec.shrink_to_fit();
      CHECK(vec.capacity() == 0);
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::shrink_to_fit, already matching
      Vector vec{resource, {T(1), T(1337), T(1), T(12), T(0), T(-1)}};
      vec.shrink_to_fit();
      CHECK(vec.capacity() == 6);
      CHECK(equal_range(vec, cuda::std::array<T, 6>{T(1), T(1337), T(1), T(12), T(0), T(-1)}));
    }

    { // cudax::vector::shrink_to_fit, shrinking
      Vector vec{resource, {T(1), T(1337), T(1), T(12), T(0), T(-1)}};
      vec.resize(2);
      CHECK(vec.capacity() == 6);
      vec.shrink_to_fit();
      CHECK(vec.capacity() == 2);
      CHECK(equal_range(vec, cuda::std::array<T, 2>{T(1), T(1337)}));
    }
  }

  SECTION("cudax::vector::reserve")
  {
    { // cudax::vector::reserve, empty noop
      Vector vec{resource};
      vec.reserve(0);
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::reserve, empty growing
      Vector vec{resource};
      vec.reserve(15);
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
      CHECK(vec.capacity() == 15);
    }

    { // cudax::vector::reserve, non-empty noop
      Vector vec{resource, {T(1), T(1337), T(1), T(12), T(0), T(-1)}};
      auto* old_ptr = vec.data();
      vec.reserve(2);
      CHECK(vec.data() == old_ptr);
      CHECK(equal_range(vec, cuda::std::array<T, 6>{T(1), T(1337), T(1), T(12), T(0), T(-1)}));
    }

    { // cudax::vector::reserve, non-empty growing
      Vector vec{resource, {T(1), T(1337), T(1), T(12), T(0), T(-1)}};
      auto* old_ptr = vec.data();
      vec.reserve(15);
      CHECK(vec.data() != old_ptr);
      CHECK(vec.capacity() == 15);
      CHECK(equal_range(vec, cuda::std::array<T, 6>{T(1), T(1337), T(1), T(12), T(0), T(-1)}));
    }
  }

#if 0 // Implement exceptions
#  ifndef TEST_HAS_NO_EXCEPTIONS
  SECTION("cudax::vector resize exceptions")
  {
    using Vector = cudax::vector<int, 42>;
    Vector too_small{};
    try
    {
      too_small.resize(1337);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }

    try
    {
      too_small.resize(1337, 5);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }

    try
    {
      too_small.reserve(1337);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }
  }
#  endif // !TEST_HAS_NO_EXCEPTIONS
#endif // Implement exceptions
}
