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
  "cudax::vector access", "[container][vector]", cuda::std::tuple<>, cuda::std::tuple<cuda::mr::host_accessible>)
{
  using Resource        = typename extract_properties<TestType>::resource;
  using Resource_ref    = typename extract_properties<TestType>::resource_ref;
  using Vector          = typename extract_properties<TestType>::vector;
  using T               = typename Vector::value_type;
  using reference       = typename Vector::reference;
  using const_reference = typename Vector::const_reference;
  using pointer         = typename Vector::pointer;
  using const_pointer   = typename Vector::const_pointer;

  Resource raw_resource{};
  Resource_ref resource{raw_resource};

  SECTION("cudax::vector::operator[]")
  {
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<Vector&>()[1ull]), reference>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const Vector&>()[1ull]), const_reference>);

    {
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      auto& res = vec[2];
      CHECK(res == T(1337));
      CHECK(cuda::std::addressof(res) - vec.data() == 2);
      res = T(4);

      auto& const_res = cuda::std::as_const(vec)[2];
      CHECK(const_res == T(4));
      CHECK(cuda::std::addressof(const_res) - vec.data() == 2);
    }
  }

  SECTION("cudax::vector::first")
  {
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<Vector&>().first()), reference>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const Vector&>().first()), const_reference>);

    {
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      auto& res = vec.first();
      CHECK(res == T(1));
      CHECK(cuda::std::addressof(res) == vec.data());
      res = T(4);

      auto& const_res = cuda::std::as_const(vec).first();
      CHECK(const_res == T(4));
      CHECK(cuda::std::addressof(const_res) == vec.data());
    }
  }

  SECTION("cudax::vector::back")
  {
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<Vector&>().back()), reference>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const Vector&>().back()), const_reference>);

    {
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      auto& res = vec.back();
      CHECK(res == T(0));
      CHECK(cuda::std::addressof(res) - vec.data() == (vec.size() - 1));
      res = T(4);

      auto& const_res = cuda::std::as_const(vec).back();
      CHECK(const_res == T(4));
      CHECK(cuda::std::addressof(const_res) - vec.data() == (vec.size() - 1));
    }
  }

  SECTION("cudax::vector::data")
  {
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<Vector&>().data()), pointer>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const Vector&>().data()), const_pointer>);

    { // Works without allocation
      Vector vec{resource};
      CHECK(vec.data() == nullptr);
      CHECK(cuda::std::as_const(vec).data() == nullptr);
    }

    { // Works with allocation
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      CHECK(vec.data() != nullptr);
      CHECK(cuda::std::as_const(vec).data() != nullptr);
      CHECK(cuda::std::as_const(vec).data() == vec.data());
    }
  }

#if 0 // Implement exceptions
#  ifndef TEST_HAS_NO_EXCEPTIONS
  SECTION("cudax::vector access exceptions")
  { // at throws std::out_of_range
    {
      using vec = cudax::vector<int, 42>;
      try
      {
        vec too_small{};
        auto res = too_small.at(5);
        unused(res);
      }
      catch (const std::out_of_range&)
      {}
      catch (...)
      {
        assert(false);
      }

      try
      {
        const vec too_small{};
        auto res = too_small.at(5);
        unused(res);
      }
      catch (const std::out_of_range&)
      {}
      catch (...)
      {
        assert(false);
      }
    }
  }
#  endif // !TEST_HAS_NO_EXCEPTIONS
#endif // Implement exceptions
}
