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

TEMPLATE_TEST_CASE("cudax::vector insert",
                   "[container][vector]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Resource     = typename extract_properties<TestType>::resource;
  using Resource_ref = typename extract_properties<TestType>::resource_ref;
  using Vector       = typename extract_properties<TestType>::vector;
  using T            = typename Vector::value_type;
  using reference    = typename Vector::reference;
  using iterator     = typename extract_properties<TestType>::iterator;

  Resource raw_resource{};
  Resource_ref resource{raw_resource};

  SECTION("cudax::vector::insert(const_iter, const T&)")
  {
    { // cudax::vector::insert(const_iter, const T&), empty without allocation
      Vector vec{resource};
      const T to_be_inserted = 3;
      auto res               = vec.insert(vec.begin(), to_be_inserted);
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_size_value(vec, 1, T(3)));
      CHECK(res == vec.begin());
    }

    { // cudax::vector::insert(const_iter, const T&), empty sufficient capacity
      Vector vec{resource, 10, T(-2)};
      vec.clear();
      const T to_be_inserted = 3;
      auto old_begin         = vec.begin();
      auto res               = vec.insert(old_begin, to_be_inserted);
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_size_value(vec, 1, T(3)));
      CHECK(res == old_begin);
    }

    { // cudax::vector::insert(const_iter, const T&), non-empty sufficient capacity
      Vector vec{resource, {T(1), T(1337), T(0), T(12), T(-1), T(1337)}};
      vec.resize(5);
      const T to_be_inserted = 42;
      auto res               = vec.insert(vec.cbegin() + 1, to_be_inserted);
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 1);
    }
    { // cudax::vector::insert(const_iter, const T&), non-empty growing
      Vector vec{resource, {T(1), T(1337), T(0), T(12), T(-1)}};
      const T to_be_inserted = 42;
      auto res               = vec.insert(vec.cbegin() + 1, to_be_inserted);
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 1);
    }

    { // cudax::vector::insert(const_iter, const T&), empty at back
      Vector vec{resource};
      const T to_be_inserted = 3;
      auto res               = vec.insert(vec.cend(), to_be_inserted);
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_size_value(vec, 1, T(3)));
      CHECK(res == vec.begin());
    }

    { // cudax::vector::insert(const_iter, const T&), non-empty at back
      Vector vec{resource, {T(1), T(42), T(1337), T(0), T(12)}};
      const T to_be_inserted = -1;
      auto res               = vec.insert(vec.cend(), to_be_inserted);
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 5);
    }
  }

  SECTION("cudax::vector::insert(const_iter, T&&)")
  {
    { // cudax::vector::insert(const_iter, T&&), empty without allocation
      Vector vec{resource};
      auto res = vec.insert(vec.begin(), T{3});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_size_value(vec, 1, T(3)));
      CHECK(res == vec.begin());
    }

    { // cudax::vector::insert(const_iter, T&&), empty sufficient capacity
      Vector vec{resource, 10, (-2)};
      vec.clear();
      auto old_begin = vec.begin();
      auto res       = vec.insert(old_begin, T{3});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_size_value(vec, 1, T(3)));
      CHECK(res == old_begin);
    }

    { // cudax::vector::insert(const_iter, T&&), non-empty sufficient capacity
      Vector vec{resource, {T(1), T(1337), T(0), T(12), T(-1), T(1337)}};
      vec.resize(5);
      auto res = vec.insert(vec.cbegin() + 1, T{42});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 1);
    }

    { // cudax::vector::insert(const_iter, T&&), non-empty growing
      Vector vec{resource, {T(1), T(1337), T(0), T(12), T(-1)}};
      auto res = vec.insert(vec.cbegin() + 1, T{42});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 1);
    }

    { // cudax::vector::insert(const_iter, T&&), empty at back
      Vector vec{resource};
      auto res = vec.insert(vec.cend(), T(3));
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_size_value(vec, 1, T(3)));
      CHECK(res == vec.begin());
    }

    { // cudax::vector::insert(const_iter, T&&), non-empty at back
      Vector vec{resource, {T(1), T(42), T(1337), T(0), T(12)}};
      auto res = vec.insert(vec.cend(), T(-1));
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 5);
    }
  }

  SECTION("cudax::vector::insert(const_iter, count, value)")
  {
    { // cudax::vector::insert(const_iter, count, value), no allocation, zero count
      Vector vec{resource};
      auto res = vec.insert(vec.begin(), 0, T{3});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(res == vec.begin());
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::insert(const_iter, count, value), with allocation, zero count
      Vector vec{resource, 10, (-2)};
      vec.clear();
      auto res = vec.insert(vec.begin(), 0, T{3});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(res == vec.begin());
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
    }

    { // cudax::vector::insert(const_iter, count, value), empty, sufficient capacity
      Vector vec{resource, 10, (-2)};
      vec.clear();
      auto old_begin = vec.begin();
      auto res       = vec.insert(old_begin, 4, T{3});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_size_value(vec, 4, T(3)));
      CHECK(res == old_begin);
    }

    { // cudax::vector::insert(const_iter, count, value), non-empty sufficient capacity
      Vector vec{resource, {T(1), T(1337), T(0), T(12), T(-1), T(1337)}};
      vec.resize(5);
      auto res = vec.insert(vec.cbegin() + 1, 1, T{42});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 1);
    }

    { // cudax::vector::insert(const_iter, count, value), non-empty growing
      Vector vec{resource, {T(1), T(1337), T(0), T(12), T(-1)}};
      auto res = vec.insert(vec.cbegin() + 1, 1, T{42});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 1);
    }

    { // cudax::vector::insert(const_iter, count, value), non-empty at back
      Vector vec{resource, {T(1), T(42), T(1337), T(0), T(12)}};
      auto res = vec.insert(vec.cend(), 1, T(-1));
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 5);
    }
  }

  SECTION("cudax::vector::insert(const_iter, iter, iter), input iterators")
  {
    const cuda::std::array<T, 6> input{T(1), T(42), T(1337), T(0), T(12), T(-1)};
    using iter = cpp17_input_iterator<const T*>;

    { // cudax::vector::insert(iter, iter, iter), no allocation empty rnge
      Vector vec{resource};
      auto res = vec.insert(vec.begin(), iter{input.begin()}, iter{input.begin()});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::insert(iter, iter, iter), from empty no allocation
      Vector vec{resource};
      auto res = vec.insert(vec.begin(), iter{input.begin()}, iter{input.end()});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin());
    }

    { // cudax::vector::insert(iter, iter, iter), from empty sufficient capacity
      Vector vec{resource, 10, T(-2)};
      vec.clear();
      auto old_begin = vec.begin();
      auto res       = vec.insert(old_begin, iter{input.begin()}, iter{input.end()});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == old_begin);
    }

    { // cudax::vector::insert(iter, iter, iter), non-empty sufficient capacity
      Vector vec{resource, {T(1), T(-1), T(-2), T(-2), T(-2), T(-2), T(-2)}};
      vec.resize(2);
      auto old_begin = vec.begin();
      auto res       = vec.insert(old_begin + 1, iter{input.begin() + 1}, iter{input.end() - 1});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == old_begin + 1);
    }

    { // cudax::vector::insert(iter, iter, iter), non-empty growing
      Vector vec{resource, {T(1), T(-1)}};
      auto res = vec.insert(vec.cbegin() + 1, iter{input.begin() + 1}, iter{input.end() - 1});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.cbegin() + 1);
    }
  }

  SECTION("cudax::vector::insert(const_iter, iter, iter), forward iterators")
  {
    const cuda::std::array<T, 6> input{T(1), T(42), T(1337), T(0), T(12), T(-1)};
    using iter = forward_iterator<const T*>;

    { // cudax::vector::insert(iter, iter, iter), no allocation empty rnge
      Vector vec{resource};
      auto res = vec.insert(vec.begin(), iter{input.begin()}, iter{input.begin()});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::insert(iter, iter, iter), no allocation
      Vector vec{resource};
      auto res = vec.insert(vec.begin(), iter{input.begin()}, iter{input.end()});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin());
    }

    { // cudax::vector::insert(iter, iter, iter), empty sufficient capacity
      Vector vec{resource, 10, T(-2)};
      vec.clear();
      auto old_begin = vec.begin();
      auto res       = vec.insert(old_begin, iter{input.begin()}, iter{input.end()});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == old_begin);
    }

    { // cudax::vector::insert(iter, iter, iter), non-empty sufficient capacity
      Vector vec{resource, {T(1), T(-1), T(-2), T(-2), T(-2), T(-2), T(-2)}};
      vec.resize(2);
      auto old_begin = vec.begin();
      auto res       = vec.insert(old_begin + 1, iter{input.begin() + 1}, iter{input.end() - 1});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == old_begin + 1);
    }

    { // cudax::vector::insert(iter, iter, iter), non-empty growing
      Vector vec{resource, {T(1), T(-1)}};
      auto res = vec.insert(vec.cbegin() + 1, iter{input.begin() + 1}, iter{input.end() - 1});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.cbegin() + 1);
    }

    { // cudax::vector::insert(iter, iter, iter), non-empty sufficient capacity, at end
      Vector vec{resource, {T(1), T(-2), T(-2), T(-2), T(-2), T(-2), T(-2)}};
      vec.resize(1);
      auto old_end = vec.end();
      auto res     = vec.insert(old_end, iter{input.begin() + 1}, iter{input.end()});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == old_end);
    }

    { // cudax::vector::insert(iter, iter, iter), non-empty growing, at end
      Vector vec{resource, {T(1)}};
      auto res = vec.insert(vec.end(), iter{input.begin() + 1}, iter{input.end()});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 1);
    }
  }

  SECTION("cudax::vector::insert(const_iter, initializer_list)")
  {
    { // cudax::vector::insert(const_iter, initializer_list), no allocation, empty input
      Vector vec{resource};
      auto res = vec.insert(vec.begin(), {});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
      CHECK(res == vec.end());
    }

    { // cudax::vector::insert(const_iter, initializer_list), with allocation, empty input
      Vector vec{resource, {T(0), T(5)}};
      vec.clear();
      auto res = vec.insert(vec.end(), {});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
      CHECK(res == vec.end());
    }

    { // cudax::vector::insert(const_iter, initializer_list), no allocation, non-empty input
      Vector vec{resource};
      auto res = vec.insert(vec.begin(), {T(1), T(42), T(1337), T(0), T(12), T(-1)});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin());
    }

    { // cudax::vector::insert(iter, iter, iter), non-empty sufficient capacity
      Vector vec{resource, {T(1), T(-1), T(-2), T(-2), T(-2), T(-2), T(-2)}};
      vec.resize(2);
      auto old_begin = vec.begin();
      auto res       = vec.insert(old_begin + 1, {T(42), T(1337), T(0), T(12)});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == old_begin + 1);
    }

    { // cudax::vector::insert(const_iter, initializer_list), non-empty sufficient capacity
      Vector vec{resource, {T(1), T(-1), T(-2), T(-2), T(-2), T(-2), T(-2)}};
      vec.resize(2);
      auto old_begin = vec.begin();
      auto res       = vec.insert(old_begin + 1, {T(42), T(1337), T(0), T(12)});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == old_begin + 1);
    }

    { // cudax::vector::insert(const_iter, initializer_list), non-empty, growing
      Vector vec{resource, {T(1), T(-1)}};
      auto res = vec.insert(vec.begin() + 1, {T(42), T(1337), T(0), T(12)});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 1);
    }

    { // cudax::vector::insert(const_iter, initializer_list), non-empty sufficient capacity, at end
      Vector vec{resource, {T(1), T(-2), T(-2), T(-2), T(-2), T(-2), T(-2)}};
      vec.resize(1);
      auto old_end = vec.end();
      auto res     = vec.insert(old_end, {T(42), T(1337), T(0), T(12), T(-1)});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == old_end);
    }

    { // cudax::vector::insert(const_iter, initializer_list), non-empty, growing, at end
      Vector vec{resource, {T(1)}};
      auto res = vec.insert(vec.end(), {T(42), T(1337), T(0), T(12), T(-1)});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 1);
    }
  }

  SECTION("cudax::vector::insert_range, input range")
  {
    { // cudax::vector::insert(const_iter, range), no allocation, empty input
      Vector vec{resource};
      auto res = vec.insert_range(vec.begin(), input_range<T, 0>{});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::insert(const_iter, range), with allocation, empty input
      Vector vec{resource, {T(0), T(5)}};
      vec.clear();
      auto res = vec.insert_range(vec.end(), input_range<T, 0>{});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
      CHECK(res == vec.end());
    }

    { // cudax::vector::insert(const_iter, range), no allocation, non-empty input
      Vector vec{resource};
      auto res = vec.insert_range(vec.begin(), input_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin());
    }

    { // cudax::vector::insert(const_iter, range), with allocation, non-empty input, sufficient capacity
      Vector vec{resource, {T(1), T(-1), T(-2), T(-2), T(-2), T(-2), T(-2)}};
      vec.resize(2);
      auto old_begin = vec.begin();
      auto res       = vec.insert_range(old_begin + 1, input_range<T, 4>{{T(42), T(1337), T(0), T(12)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == old_begin + 1);
    }

    { // cudax::vector::insert(const_iter, range), with allocation, non-empty input, growing
      Vector vec{resource, {T(1), T(-1)}};
      auto res = vec.insert_range(vec.begin() + 1, input_range<T, 4>{{T(42), T(1337), T(0), T(12)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 1);
    }

    { // cudax::vector::insert(const_iter, range), with allocation, non-empty input, sufficient capacity, at end
      Vector vec{resource, {T(1), T(42), T(-2), T(-2), T(-2), T(-2), T(-2)}};
      vec.resize(2);
      auto old_end = vec.end();
      auto res     = vec.insert_range(old_end, input_range<T, 4>{{T(1337), T(0), T(12), T(-1)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == old_end);
    }

    { // cudax::vector::insert(const_iter, range), with allocation, non-empty input, growing, at end
      Vector vec{resource, {T(1), T(42)}};
      auto res = vec.insert_range(vec.end(), input_range<T, 4>{{T(1337), T(0), T(12), T(-1)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 2);
    }
  }

  SECTION("cudax::template::append_range, input_range")
  {
    { // cudax::vector::append_range(range), no allocation, empty input
      Vector vec{resource};
      vec.append_range(input_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::append_range(range), with allocation, empty input
      Vector vec{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      auto old_begin = vec.begin();
      vec.append_range(input_range<T, 0>{});
      CHECK(equal_range(vec));
      CHECK(vec.begin() == old_begin);
    }

    { // cudax::vector::append_range(range), with allocation, non-empty input, sufficient capacity
      Vector vec{resource, {T(1), T(42), T(1337), T(0), T(-2), T(-2), T(-2), T(-2), T(-2)}};
      vec.resize(4);
      auto old_begin = vec.begin();
      vec.append_range(input_range<T, 2>{{T(12), T(-1)}});
      CHECK(equal_range(vec));
      CHECK(vec.begin() == old_begin);
    }

    { // cudax::vector::append_range(range), with allocation, non-empty input, growing
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.append_range(input_range<T, 2>{{T(12), T(-1)}});
      CHECK(equal_range(vec));
    }
  }

  SECTION("cudax::vector::insert_range, uncommon range")
  {
    { // cudax::vector::insert(const_iter, range), no allocation, empty input
      Vector vec{resource};
      auto res = vec.insert_range(vec.begin(), uncommon_range<T, 0>{});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::insert(const_iter, range), with allocation, empty input
      Vector vec{resource, {T(0), T(5)}};
      vec.clear();
      auto res = vec.insert_range(vec.end(), uncommon_range<T, 0>{});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
      CHECK(res == vec.end());
    }

    { // cudax::vector::insert(const_iter, range), no allocation, non-empty input
      Vector vec{resource};
      auto res = vec.insert_range(vec.begin(), uncommon_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin());
    }

    { // cudax::vector::insert(const_iter, range), with allocation, non-empty input, sufficient capacity
      Vector vec{resource, {T(1), T(-1), T(-2), T(-2), T(-2), T(-2), T(-2)}};
      vec.resize(2);
      auto old_begin = vec.begin();
      auto res       = vec.insert_range(old_begin + 1, uncommon_range<T, 4>{{T(42), T(1337), T(0), T(12)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == old_begin + 1);
    }

    { // cudax::vector::insert(const_iter, range), with allocation, non-empty input, growing
      Vector vec{resource, {T(1), T(-1)}};
      auto res = vec.insert_range(vec.begin() + 1, uncommon_range<T, 4>{{T(42), T(1337), T(0), T(12)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 1);
    }

    { // cudax::vector::insert(const_iter, range), with allocation, non-empty input, sufficient capacity, at end
      Vector vec{resource, {T(1), T(42), T(-2), T(-2), T(-2), T(-2), T(-2)}};
      vec.resize(2);
      auto old_end = vec.end();
      auto res     = vec.insert_range(old_end, uncommon_range<T, 4>{{T(1337), T(0), T(12), T(-1)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == old_end);
    }

    { // cudax::vector::insert(const_iter, range), with allocation, non-empty input, growing, at end
      Vector vec{resource, {T(1), T(42)}};
      auto res = vec.insert_range(vec.end(), uncommon_range<T, 4>{{T(1337), T(0), T(12), T(-1)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 2);
    }
  }

  SECTION("cudax::template::append_range, uncommon range")
  {
    { // cudax::vector::append_range(range), no allocation, empty input
      Vector vec{resource};
      vec.append_range(uncommon_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::append_range(range), with allocation, empty input
      Vector vec{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      auto old_begin = vec.begin();
      vec.append_range(uncommon_range<T, 0>{});
      CHECK(equal_range(vec));
      CHECK(vec.begin() == old_begin);
    }

    { // cudax::vector::append_range(range), with allocation, non-empty input, sufficient capacity
      Vector vec{resource, {T(1), T(42), T(1337), T(0), T(-2), T(-2), T(-2), T(-2), T(-2)}};
      vec.resize(4);
      auto old_begin = vec.begin();
      vec.append_range(uncommon_range<T, 2>{{T(12), T(-1)}});
      CHECK(equal_range(vec));
      CHECK(vec.begin() == old_begin);
    }

    { // cudax::vector::append_range(range), with allocation, non-empty input, growing
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.append_range(uncommon_range<T, 2>{{T(12), T(-1)}});
      CHECK(equal_range(vec));
    }
  }

  SECTION("cudax::vector::insert_range, sized uncommon range")
  {
    { // cudax::vector::insert(const_iter, range), no allocation, empty input
      Vector vec{resource};
      auto res = vec.insert_range(vec.begin(), sized_uncommon_range<T, 0>{});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::insert(const_iter, range), with allocation, empty input
      Vector vec{resource, {T(0), T(5)}};
      vec.clear();
      auto res = vec.insert_range(vec.end(), sized_uncommon_range<T, 0>{});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
      CHECK(res == vec.end());
    }

    { // cudax::vector::insert(const_iter, range), no allocation, non-empty input
      Vector vec{resource};
      auto res = vec.insert_range(vec.begin(), sized_uncommon_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin());
    }

    { // cudax::vector::insert(const_iter, range), with allocation, non-empty input, sufficient capacity
      Vector vec{resource, {T(1), T(-1), T(-2), T(-2), T(-2), T(-2), T(-2)}};
      vec.resize(2);
      auto old_begin = vec.begin();
      auto res       = vec.insert_range(old_begin + 1, sized_uncommon_range<T, 4>{{T(42), T(1337), T(0), T(12)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == old_begin + 1);
    }

    { // cudax::vector::insert(const_iter, range), with allocation, non-empty input, growing
      Vector vec{resource, {T(1), T(-1)}};
      auto res = vec.insert_range(vec.begin() + 1, sized_uncommon_range<T, 4>{{T(42), T(1337), T(0), T(12)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 1);
    }

    { // cudax::vector::insert(const_iter, range), with allocation, non-empty input, sufficient capacity, at end
      Vector vec{resource, {T(1), T(42), T(-2), T(-2), T(-2), T(-2), T(-2)}};
      vec.resize(2);
      auto old_end = vec.end();
      auto res     = vec.insert_range(old_end, sized_uncommon_range<T, 4>{{T(1337), T(0), T(12), T(-1)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == old_end);
    }

    { // cudax::vector::insert(const_iter, range), with allocation, non-empty input, growing, at end
      Vector vec{resource, {T(1), T(42)}};
      auto res = vec.insert_range(vec.end(), sized_uncommon_range<T, 4>{{T(1337), T(0), T(12), T(-1)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 2);
    }
  }

  SECTION("cudax::template::append_range, sized uncommon range")
  {
    { // cudax::vector::append_range(range), no allocation, empty input
      Vector vec{resource};
      vec.append_range(sized_uncommon_range<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::append_range(range), with allocation, empty input
      Vector vec{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      auto old_begin = vec.begin();
      vec.append_range(sized_uncommon_range<T, 0>{});
      CHECK(equal_range(vec));
      CHECK(vec.begin() == old_begin);
    }

    { // cudax::vector::append_range(range), with allocation, non-empty input, sufficient capacity
      Vector vec{resource, {T(1), T(42), T(1337), T(0), T(-2), T(-2), T(-2), T(-2), T(-2)}};
      vec.resize(4);
      auto old_begin = vec.begin();
      vec.append_range(sized_uncommon_range<T, 2>{{T(12), T(-1)}});
      CHECK(equal_range(vec));
      CHECK(vec.begin() == old_begin);
    }

    { // cudax::vector::append_range(range), with allocation, non-empty input, growing
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.append_range(sized_uncommon_range<T, 2>{{T(12), T(-1)}});
      CHECK(equal_range(vec));
    }
  }

  SECTION("cudax::vector::insert_range, random_access range")
  {
    { // cudax::vector::insert(const_iter, range), no allocation, empty input
      Vector vec{resource};
      auto res = vec.insert_range(vec.begin(), cuda::std::array<T, 0>{});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::insert(const_iter, range), with allocation, empty input
      Vector vec{resource, {T(0), T(5)}};
      vec.clear();
      auto res = vec.insert_range(vec.end(), cuda::std::array<T, 0>{});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(vec.empty());
      CHECK(vec.data() != nullptr);
      CHECK(res == vec.end());
    }

    { // cudax::vector::insert(const_iter, range), no allocation, non-empty input
      Vector vec{resource};
      auto res = vec.insert_range(vec.begin(), cuda::std::array<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin());
    }

    { // cudax::vector::insert(const_iter, range), with allocation, non-empty input, sufficient capacity
      Vector vec{resource, {T(1), T(-1), T(-2), T(-2), T(-2), T(-2), T(-2)}};
      vec.resize(2);
      auto old_begin = vec.begin();
      auto res       = vec.insert_range(old_begin + 1, cuda::std::array<T, 4>{{T(42), T(1337), T(0), T(12)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == old_begin + 1);
    }

    { // cudax::vector::insert(const_iter, range), with allocation, non-empty input, growing
      Vector vec{resource, {T(1), T(-1)}};
      auto res = vec.insert_range(vec.begin() + 1, cuda::std::array<T, 4>{{T(42), T(1337), T(0), T(12)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 1);
    }

    { // cudax::vector::insert(const_iter, range), with allocation, non-empty input, sufficient capacity, at end
      Vector vec{resource, {T(1), T(42), T(-2), T(-2), T(-2), T(-2), T(-2)}};
      vec.resize(2);
      auto old_end = vec.end();
      auto res     = vec.insert_range(old_end, cuda::std::array<T, 4>{{T(1337), T(0), T(12), T(-1)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == old_end);
    }

    { // cudax::vector::insert(const_iter, range), with allocation, non-empty input, growing, at end
      Vector vec{resource, {T(1), T(42)}};
      auto res = vec.insert_range(vec.end(), sized_uncommon_range<T, 4>{{T(1337), T(0), T(12), T(-1)}});
      static_assert(cuda::std::is_same<decltype(res), iterator>::value, "");
      CHECK(equal_range(vec));
      CHECK(res == vec.begin() + 2);
    }
  }

  SECTION("cudax::template::append_range, random_access range")
  {
    { // cudax::vector::append_range(range), no allocation, empty input
      Vector vec{resource};
      vec.append_range(cuda::std::array<T, 0>{});
      CHECK(vec.empty());
      CHECK(vec.data() == nullptr);
    }

    { // cudax::vector::append_range(range), with allocation, empty input
      Vector vec{resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      auto old_begin = vec.begin();
      vec.append_range(cuda::std::array<T, 0>{});
      CHECK(equal_range(vec));
      CHECK(vec.begin() == old_begin);
    }

    { // cudax::vector::append_range(range), with allocation, non-empty input, sufficient capacity
      Vector vec{resource, {T(1), T(42), T(1337), T(0), T(-2), T(-2), T(-2), T(-2), T(-2)}};
      vec.resize(4);
      auto old_begin = vec.begin();
      vec.append_range(cuda::std::array<T, 2>{{T(12), T(-1)}});
      CHECK(equal_range(vec));
      CHECK(vec.begin() == old_begin);
    }

    { // cudax::vector::append_range(range), with allocation, non-empty input, growing
      Vector vec{resource, {T(1), T(42), T(1337), T(0)}};
      vec.append_range(cuda::std::array<T, 2>{{T(12), T(-1)}});
      CHECK(equal_range(vec));
    }
  }

#if 0 // Implement exceptions
#  ifndef TEST_HAS_NO_EXCEPTIONS
{ // insert throws std::bad_alloc
  using Vector = cudax::vector<int, 2>;
  Vector too_small{1, 2};

  try
  {
    const int input = 5;
    too_small.insert(too_small.begin(), input);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    CHECK(false);
  }

  try
  {
    too_small.insert(too_small.begin(), 1);
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    CHECK(false);
  }

  try
  {
    too_small.insert(too_small.begin(), 5, 42);
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
    cuda::std::array<int, 3> input{42, 3, 1337};
    too_small.insert(too_small.begin(), iter{input.begin()}, iter{input.end()});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    CHECK(false);
  }

  try
  {
    cuda::std::array<int, 3> input{42, 3, 1337};
    too_small.insert(too_small.begin(), input.begin(), input.end());
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    CHECK(false);
  }

  try
  {
    too_small.insert(too_small.begin(), {42, 3, 1337});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    CHECK(false);
  }

  try
  {
    too_small.insert_range(too_small.begin(), input_range<int, 3>{42, 3, 1337});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    CHECK(false);
  }

  try
  {
    too_small.insert_range(too_small.begin(), uncommon_range<int, 3>{42, 3, 1337});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    CHECK(false);
  }

  try
  {
    too_small.insert_range(too_small.begin(), sized_uncommon_range<int, 3>{42, 3, 1337});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    CHECK(false);
  }

  try
  {
    too_small.insert_range(too_small.begin(), cuda::std::array<int, 3>{42, 3, 1337});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    CHECK(false);
  }

  try
  {
    too_small.append_range(input_range<int, 3>{42, 3, 1337});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    CHECK(false);
  }

  try
  {
    too_small.append_range(uncommon_range<int, 3>{42, 3, 1337});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    CHECK(false);
  }

  try
  {
    too_small.append_range(sized_uncommon_range<int, 3>{42, 3, 1337});
  }
  catch (const std::bad_alloc&)
  {}
  catch (...)
  {
    CHECK(false);
  }

  try
  {
    too_small.append_range(cuda::std::array<int, 3>{42, 3, 1337});
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
