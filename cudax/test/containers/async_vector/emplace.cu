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

TEMPLATE_TEST_CASE("cudax::async_vector emplace",
                   "[container][async_vector]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Env       = typename extract_properties<TestType>::env;
  using Resource  = typename extract_properties<TestType>::resource;
  using Vector    = typename extract_properties<TestType>::async_vector;
  using T         = typename Vector::value_type;
  using reference = typename Vector::reference;
  using iterator  = typename extract_properties<TestType>::iterator;

  cudax::stream stream{};
  Env env{Resource{}, stream};

  SECTION("cudax::async_vector::emplace(const_iter, args...)")
  {
    { // cudax::async_vector::emplace(const_iter, args...), empty without allocation
      Vector vec{env};
      const auto res = vec.emplace(vec.begin(), 3);
      static_assert(cuda::std::is_same<decltype(res), const iterator>::value, "");
      CHECK(equal_size_value(vec, 1, T(3)));
      CHECK(res == vec.begin());
    }

    { // cudax::async_vector::emplace(const_iter, args...), empty with allocation
      Vector vec{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      vec.clear();
      auto old_begin = vec.begin();
      const auto res = vec.emplace(old_begin, 3);
      static_assert(cuda::std::is_same<decltype(res), const iterator>::value, "");
      CHECK(equal_size_value(vec, 1, T(3)));
      CHECK(res == old_begin);
    }

    { // cudax::async_vector::emplace(const_iter, args...), sufficient capacity
      Vector vec{env, {T(1), T(42), T(0), T(12), T(-1), T(1337)}};
      vec.resize(5);
      auto old_begin = vec.begin();
      const auto res = vec.emplace(old_begin + 2, 1337);
      static_assert(cuda::std::is_same<decltype(res), const iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == old_begin + 2);
    }

    { // cudax::async_vector::emplace(const_iter, args...), growing
      Vector vec{env, {T(1), T(42), T(0), T(12), T(-1)}};
      auto old_begin = vec.begin();
      const auto res = vec.emplace(old_begin + 2, 1337);
      static_assert(cuda::std::is_same<decltype(res), const iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.cbegin() + 2);
    }
  }

  SECTION("cudax::async_vector::emplace_back(args...)")
  {
    { // cudax::async_vector::emplace_back(args...), empty without allocation
      Vector vec{env};
      decltype(auto) res = vec.emplace_back(3);
      static_assert(cuda::std::is_same<decltype(res), reference>::value, "");
      CHECK(equal_size_value(vec, 1, T(3)));
      CHECK(cuda::std::addressof(res) == cuda::std::addressof(vec.back())); // cannot compare values for device only
    }

    { // cudax::async_vector::emplace_back(args...), empty with allocation
      Vector vec{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      vec.clear();
      decltype(auto) res = vec.emplace_back(3);
      static_assert(cuda::std::is_same<decltype(res), reference>::value, "");
      CHECK(equal_size_value(vec, 1, T(3)));
      CHECK(cuda::std::addressof(res) == cuda::std::addressof(vec.back())); // cannot compare values for device only
    }

    { // cudax::async_vector::emplace_back(args...), sufficient capacity
      Vector vec{env, {T(1), T(42), T(1337), T(0), T(12), T(42)}};
      vec.resize(5);
      decltype(auto) res = vec.emplace_back(-1);
      static_assert(cuda::std::is_same<decltype(res), reference>::value, "");
      CHECK(equal_range(vec));
      CHECK(cuda::std::addressof(res) == cuda::std::addressof(vec.back())); // cannot compare values for device only
    }

    { // cudax::async_vector::emplace_back(args...), growing
      Vector vec{env, {T(1), T(42), T(1337), T(0), T(12)}};
      decltype(auto) res = vec.emplace_back(-1);
      static_assert(cuda::std::is_same<decltype(res), reference>::value, "");
      CHECK(equal_range(vec));
      CHECK(cuda::std::addressof(res) == cuda::std::addressof(vec.back())); // cannot compare values for device only
    }
  }

  SECTION("cudax::async_vector::push_back(const T&)")
  {
    { // cudax::async_vector::push_back(const T&), empty without allocation
      const T input{3};
      Vector vec{env};
      decltype(auto) res = vec.push_back(input);
      static_assert(cuda::std::is_same<decltype(res), reference>::value, "");
      CHECK(equal_size_value(vec, 1, T(3)));
      CHECK(cuda::std::addressof(res) == cuda::std::addressof(vec.back()));
    }

    { // cudax::async_vector::push_back(const T&), empty with allocation
      const T input{3};
      Vector vec{env, {T(1), T(42), T(1337), T(0), T(12), T(42)}};
      vec.clear();
      decltype(auto) res = vec.push_back(input);
      static_assert(cuda::std::is_same<decltype(res), reference>::value, "");
      CHECK(equal_size_value(vec, 1, T(3)));
      CHECK(cuda::std::addressof(res) == cuda::std::addressof(vec.back()));
    }

    { // cudax::async_vector::push_back(const T&), sufficient capacity
      const T input{-1};
      Vector vec{env, {T(1), T(42), T(1337), T(0), T(12), T(42)}};
      vec.resize(5);
      decltype(auto) res = vec.push_back(input);
      static_assert(cuda::std::is_same<decltype(res), reference>::value, "");
      CHECK(equal_range(vec));
      CHECK(cuda::std::addressof(res) == cuda::std::addressof(vec.back()));
    }

    { // cudax::async_vector::push_back(const T&), growing
      const T input{-1};
      Vector vec{env, {T(1), T(42), T(1337), T(0), T(12)}};
      decltype(auto) res = vec.push_back(input);
      static_assert(cuda::std::is_same<decltype(res), reference>::value, "");
      CHECK(equal_range(vec));
      CHECK(cuda::std::addressof(res) == cuda::std::addressof(vec.back()));
    }
  }

  SECTION("cudax::async_vector::push_back(T&&)")
  {
    { // cudax::async_vector::push_back(T&&), empty without allocation
      Vector vec{env};
      decltype(auto) res = vec.push_back(T{3});
      static_assert(cuda::std::is_same<decltype(res), reference>::value, "");
      CHECK(equal_size_value(vec, 1, T(3)));
      CHECK(cuda::std::addressof(res) == cuda::std::addressof(vec.back()));
    }

    { // cudax::async_vector::push_back(T&&), empty with allocation
      Vector vec{env, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      vec.clear();
      decltype(auto) res = vec.push_back(T{3});
      static_assert(cuda::std::is_same<decltype(res), reference>::value, "");
      CHECK(equal_size_value(vec, 1, T(3)));
      CHECK(cuda::std::addressof(res) == cuda::std::addressof(vec.back()));
    }

    { // cudax::async_vector::push_back(T&&), sufficient capacity
      Vector vec{env, {T(1), T(42), T(1337), T(0), T(12), T(42)}};
      vec.resize(5);
      decltype(auto) res = vec.push_back(T{-1});
      static_assert(cuda::std::is_same<decltype(res), reference>::value, "");
      CHECK(equal_range(vec));
      CHECK(cuda::std::addressof(res) == cuda::std::addressof(vec.back()));
    }

    { // cudax::async_vector::push_back(T&&), growing
      Vector vec{env, {T(1), T(42), T(1337), T(0), T(12)}};
      decltype(auto) res = vec.push_back(T{-1});
      static_assert(cuda::std::is_same<decltype(res), reference>::value, "");
      CHECK(equal_range(vec));
      CHECK(cuda::std::addressof(res) == cuda::std::addressof(vec.back()));
    }
  }

#if 0 // Implement exceptions
#  ifndef TEST_HAS_NO_EXCEPTIONS
  SECTION("cudax::async_vector::emplace Exception handling") {
  {
    empty_vec empty{};
    try
    {
      auto emplace = empty.emplace_back(5);
      unused(emplace);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }

    try
    {
      const int input  =5;
      auto push_back_lvalue = empty.push_back(input);
      unused(push_back_lvalue);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }

    try
    {
      auto push_back_rvalue = empty.push_back(5);
      unused(push_back_rvalue);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }
  }

  using small_vec = cudax::async_vector<int, 5>;
  {
    small_vec full{0, 1, 2, 3, 4};
    try
    {
      auto emplace = full.emplace_back(5);
      unused(emplace);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }

    try
    {
      const int input  =5;
      auto push_back_lvalue = full.push_back(input);
      unused(push_back_lvalue);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }

    try
    {
      auto push_back_rvalue = full.push_back(5);
      unused(push_back_rvalue);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CHECK(false);
    }
  }
  }
#  endif // !TEST_HAS_NO_EXCEPTIONS
#endif // Implement exceptions
}
