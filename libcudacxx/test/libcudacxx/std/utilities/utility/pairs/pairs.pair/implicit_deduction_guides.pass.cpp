//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides
// UNSUPPORTED: msvc

// UNSUPPORTED: nvcc-10.3, nvcc-11.0, nvcc-11.1, nvcc-11.2, nvcc-11.3, nvcc-11.4

// GCC's implementation of class template deduction is still immature and runs
// into issues with libc++. However GCC accepts this code when compiling
// against libstdc++.
// XFAIL: gcc-4.8, gcc-5, gcc-6, gcc-7, gcc-8, gcc-9, gcc-10, gcc-11

// Currently broken with Clang + NVCC.
// XFAIL: clang-6, clang-7

// <utility>

// Test that the constructors offered by cuda::std::pair are formulated
// so they're compatible with implicit deduction guides, or if that's not
// possible that they provide explicit guides to make it work.

#include <cuda/std/utility>
// cuda/std/memory not supported
// #include <cuda/std/memory>
// cuda::std::string not supported
// #include <cuda/std/string>
#include <cuda/std/cassert>

#include "archetypes.h"
#include "test_macros.h"

// Overloads
// ---------------
// (1)  pair(const T1&, const T2&) -> pair<T1, T2>
// (2)  explicit pair(const T1&, const T2&) -> pair<T1, T2>
// (3)  pair(pair const& t) -> decltype(t)
// (4)  pair(pair&& t) -> decltype(t)
// (5)  pair(pair<U1, U2> const&) -> pair<U1, U2>
// (6)  explicit pair(pair<U1, U2> const&) -> pair<U1, U2>
// (7)  pair(pair<U1, U2> &&) -> pair<U1, U2>
// (8)  explicit pair(pair<U1, U2> &&) -> pair<U1, U2>
int main(int, char**)
{
  using E = ExplicitTestTypes::TestType;
  static_assert(!cuda::std::is_convertible<E const&, E>::value, "");
  { // Testing (1)
    int const x = 42;
    cuda::std::pair t1("abc", x);
    ASSERT_SAME_TYPE(decltype(t1), cuda::std::pair<const char*, int>);
    unused(t1);
  }
  { // Testing (2)
    cuda::std::pair p1(E{}, 42);
    ASSERT_SAME_TYPE(decltype(p1), cuda::std::pair<E, int>);
    unused(p1);

    const E t{};
    cuda::std::pair p2(t, E{});
    ASSERT_SAME_TYPE(decltype(p2), cuda::std::pair<E, E>);
  }
  { // Testing (3, 5)
    cuda::std::pair<double, decltype(nullptr)> const p(0.0, nullptr);
    cuda::std::pair p1(p);
    unused(p1);
    ASSERT_SAME_TYPE(decltype(p1), cuda::std::pair<double, decltype(nullptr)>);
  }
  { // Testing (3, 6)
    cuda::std::pair<E, decltype(nullptr)> const p(E{}, nullptr);
    cuda::std::pair p1(p);
    unused(p1);
    ASSERT_SAME_TYPE(decltype(p1), cuda::std::pair<E, decltype(nullptr)>);
  }
  // cuda::std::string not supported
  /*
  { // Testing (4, 7)
    cuda::std::pair<cuda::std::string, void*> p("abc", nullptr);
    cuda::std::pair p1(cuda::std::move(p));
    ASSERT_SAME_TYPE(decltype(p1), cuda::std::pair<cuda::std::string, void*>);
  }
  { // Testing (4, 8)
    cuda::std::pair<cuda::std::string, E> p("abc", E{});
    cuda::std::pair p1(cuda::std::move(p));
    ASSERT_SAME_TYPE(decltype(p1), cuda::std::pair<cuda::std::string, E>);
  }
  */
  return 0;
}
