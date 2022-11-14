//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11

// <span>

// template<size_t N>
//     constexpr span(array<value_type, N>& arr) noexcept;
// template<size_t N>
//     constexpr span(const array<value_type, N>& arr) noexcept;
//
// Remarks: These constructors shall not participate in overload resolution unless:
//   — extent == dynamic_extent || N == extent is true, and
//   — remove_pointer_t<decltype(data(arr))>(*)[] is convertible to ElementType(*)[].
//


#include <cuda/std/span>
#include <cuda/std/array>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__
void checkCV()
{
    cuda::std::array<int, 3> arr  = {1,2,3};
//  STL says these are not cromulent
//  std::array<const int,3> carr = {4,5,6};
//  std::array<volatile int, 3> varr = {7,8,9};
//  std::array<const volatile int, 3> cvarr = {1,3,5};

//  Types the same (dynamic sized)
    {
    cuda::std::span<               int> s1{  arr};    // a cuda::std::span<               int> pointing at int.
    }

//  Types the same (static sized)
    {
    cuda::std::span<               int,3> s1{  arr};  // a cuda::std::span<               int> pointing at int.
    }


//  types different (dynamic sized)
    {
    cuda::std::span<const          int> s1{ arr};     // a cuda::std::span<const          int> pointing at int.
    cuda::std::span<      volatile int> s2{ arr};     // a cuda::std::span<      volatile int> pointing at int.
    cuda::std::span<      volatile int> s3{ arr};     // a cuda::std::span<      volatile int> pointing at const int.
    cuda::std::span<const volatile int> s4{ arr};     // a cuda::std::span<const volatile int> pointing at int.
    }

//  types different (static sized)
    {
    cuda::std::span<const          int,3> s1{ arr};   // a cuda::std::span<const          int> pointing at int.
    cuda::std::span<      volatile int,3> s2{ arr};   // a cuda::std::span<      volatile int> pointing at int.
    cuda::std::span<      volatile int,3> s3{ arr};   // a cuda::std::span<      volatile int> pointing at const int.
    cuda::std::span<const volatile int,3> s4{ arr};   // a cuda::std::span<const volatile int> pointing at int.
    }
}

template <typename T, typename U>
__host__ __device__
TEST_CONSTEXPR_CXX17 bool testConstructorArray() {
  cuda::std::array<U, 2> val = {U(), U()};
  ASSERT_NOEXCEPT(cuda::std::span<T>{val});
  ASSERT_NOEXCEPT(cuda::std::span<T, 2>{val});
  cuda::std::span<T> s1{val};
  cuda::std::span<T, 2> s2{val};
  return s1.data() == &val[0] && s1.size() == 2 && s2.data() == &val[0] &&
         s2.size() == 2;
}

template <typename T, typename U>
__host__ __device__
TEST_CONSTEXPR_CXX17 bool testConstructorConstArray() {
  const cuda::std::array<U, 2> val = {U(), U()};
  ASSERT_NOEXCEPT(cuda::std::span<const T>{val});
  ASSERT_NOEXCEPT(cuda::std::span<const T, 2>{val});
  cuda::std::span<const T> s1{val};
  cuda::std::span<const T, 2> s2{val};
  return s1.data() == &val[0] && s1.size() == 2 && s2.data() == &val[0] &&
         s2.size() == 2;
}

template <typename T>
__host__ __device__
TEST_CONSTEXPR_CXX17 bool testConstructors() {
  STATIC_ASSERT_CXX17((testConstructorArray<T, T>()));
  STATIC_ASSERT_CXX17((testConstructorArray<const T, const T>()));
  STATIC_ASSERT_CXX17((testConstructorArray<const T, T>()));
  STATIC_ASSERT_CXX17((testConstructorConstArray<T, T>()));
  STATIC_ASSERT_CXX17((testConstructorConstArray<const T, const T>()));
  STATIC_ASSERT_CXX17((testConstructorConstArray<const T, T>()));

  return testConstructorArray<T, T>() &&
         testConstructorArray<const T, const T>() &&
         testConstructorArray<const T, T>() &&
         testConstructorConstArray<T, T>() &&
         testConstructorConstArray<const T, const T>() &&
         testConstructorConstArray<const T, T>();
}

struct A{};

int main(int, char**)
{
    assert(testConstructors<int>());
    assert(testConstructors<long>());
    assert(testConstructors<double>());
    assert(testConstructors<A>());

    assert(testConstructors<int*>());
    assert(testConstructors<const int*>());

    checkCV();

    return 0;
}
