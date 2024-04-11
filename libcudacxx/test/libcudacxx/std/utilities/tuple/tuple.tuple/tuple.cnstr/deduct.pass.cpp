//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides
// UNSUPPORTED: apple-clang-9
// UNSUPPORTED: msvc
// UNSUPPORTED: nvcc-10.3, nvcc-11.0, nvcc-11.1, nvcc-11.2, nvcc-11.3, nvcc-11.4

// GCC's implementation of class template deduction is still immature and runs
// into issues with libc++. However GCC accepts this code when compiling
// against libstdc++.
// XFAIL: gcc-5, gcc-6, gcc-7, gcc-10

// Currently broken with Clang + NVCC.
// XFAIL: clang-6, clang-7

// <cuda/std/tuple>

// Test that the constructors offered by cuda::std::tuple are formulated
// so they're compatible with implicit deduction guides, or if that's not
// possible that they provide explicit guides to make it work.

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "archetypes.h"
#include "test_macros.h"

// Overloads
//  using A = Allocator
//  using AT = cuda::std::allocator_arg_t
// ---------------
// (1)  tuple(const Types&...) -> tuple<Types...>
// (2)  tuple(pair<T1, T2>) -> tuple<T1, T2>;
// (3)  explicit tuple(const Types&...) -> tuple<Types...>
// (4)  tuple(AT, A const&, Types const&...) -> tuple<Types...>
// (5)  explicit tuple(AT, A const&, Types const&...) -> tuple<Types...>
// (6)  tuple(AT, A, pair<T1, T2>) -> tuple<T1, T2>
// (7)  tuple(tuple const& t) -> decltype(t)
// (8)  tuple(tuple&& t) -> decltype(t)
// (9)  tuple(AT, A const&, tuple const& t) -> decltype(t)
// (10) tuple(AT, A const&, tuple&& t) -> decltype(t)
__host__ __device__ void test_primary_template()
{
  // cuda::std::allocator not supported
  // const cuda::std::allocator<int> A;
  const auto AT = cuda::std::allocator_arg;
  unused(AT);
  { // Testing (1)
    int x = 101;
    cuda::std::tuple t1(42);
    ASSERT_SAME_TYPE(decltype(t1), cuda::std::tuple<int>);
    cuda::std::tuple t2(x, 0.0, nullptr);
    ASSERT_SAME_TYPE(decltype(t2), cuda::std::tuple<int, double, decltype(nullptr)>);
    unused(t1);
    unused(t2);
  }
  { // Testing (2)
    cuda::std::pair<int, char> p1(1, 'c');
    cuda::std::tuple t1(p1);
    ASSERT_SAME_TYPE(decltype(t1), cuda::std::tuple<int, char>);

    cuda::std::pair<int, cuda::std::tuple<char, long, void*>> p2(
      1, cuda::std::tuple<char, long, void*>('c', 3l, nullptr));
    cuda::std::tuple t2(p2);
    ASSERT_SAME_TYPE(decltype(t2), cuda::std::tuple<int, cuda::std::tuple<char, long, void*>>);

    int i = 3;
    cuda::std::pair<cuda::std::reference_wrapper<int>, char> p3(cuda::std::ref(i), 'c');
    cuda::std::tuple t3(p3);
    ASSERT_SAME_TYPE(decltype(t3), cuda::std::tuple<cuda::std::reference_wrapper<int>, char>);

    cuda::std::pair<int&, char> p4(i, 'c');
    cuda::std::tuple t4(p4);
    ASSERT_SAME_TYPE(decltype(t4), cuda::std::tuple<int&, char>);

    cuda::std::tuple t5(cuda::std::pair<int, char>(1, 'c'));
    ASSERT_SAME_TYPE(decltype(t5), cuda::std::tuple<int, char>);
    unused(t5);
  }
  { // Testing (3)
    using T = ExplicitTestTypes::TestType;
    static_assert(!cuda::std::is_convertible<T const&, T>::value, "");

    cuda::std::tuple t1(T{});
    ASSERT_SAME_TYPE(decltype(t1), cuda::std::tuple<T>);

#if defined(__GNUC__) && (__GNUC__ < 11)
    const T v{};
    cuda::std::tuple t2(T{}, 101l, v);
    ASSERT_SAME_TYPE(decltype(t2), cuda::std::tuple<T, long, T>);
#endif
  }
  // cuda::std::allocator not supported
  /*
  { // Testing (4)
    int x = 101;
    cuda::std::tuple t1(AT, A, 42);
    ASSERT_SAME_TYPE(decltype(t1), cuda::std::tuple<int>);

    cuda::std::tuple t2(AT, A, 42, 0.0, x);
    ASSERT_SAME_TYPE(decltype(t2), cuda::std::tuple<int, double, int>);
  }
  { // Testing (5)
    using T = ExplicitTestTypes::TestType;
    static_assert(!cuda::std::is_convertible<T const&, T>::value, "");

    cuda::std::tuple t1(AT, A, T{});
    ASSERT_SAME_TYPE(decltype(t1), cuda::std::tuple<T>);

    const T v{};
    cuda::std::tuple t2(AT, A, T{}, 101l, v);
    ASSERT_SAME_TYPE(decltype(t2), cuda::std::tuple<T, long, T>);
  }
  { // Testing (6)
    cuda::std::pair<int, char> p1(1, 'c');
    cuda::std::tuple t1(AT, A, p1);
    ASSERT_SAME_TYPE(decltype(t1), cuda::std::tuple<int, char>);

    cuda::std::pair<int, cuda::std::tuple<char, long, void*>> p2(1, cuda::std::tuple<char, long, void*>('c', 3l,
  nullptr)); cuda::std::tuple t2(AT, A, p2); ASSERT_SAME_TYPE(decltype(t2), cuda::std::tuple<int, cuda::std::tuple<char,
  long, void*>>);

    int i = 3;
    cuda::std::pair<cuda::std::reference_wrapper<int>, char> p3(cuda::std::ref(i), 'c');
    cuda::std::tuple t3(AT, A, p3);
    ASSERT_SAME_TYPE(decltype(t3), cuda::std::tuple<cuda::std::reference_wrapper<int>, char>);

    cuda::std::pair<int&, char> p4(i, 'c');
    cuda::std::tuple t4(AT, A, p4);
    ASSERT_SAME_TYPE(decltype(t4), cuda::std::tuple<int&, char>);

    cuda::std::tuple t5(AT, A, cuda::std::pair<int, char>(1, 'c'));
    ASSERT_SAME_TYPE(decltype(t5), cuda::std::tuple<int, char>);
  }
  */
  { // Testing (7)
    using Tup = cuda::std::tuple<int, decltype(nullptr)>;
    const Tup t(42, nullptr);

    cuda::std::tuple t1(t);
    ASSERT_SAME_TYPE(decltype(t1), Tup);
    unused(t1);
  }
  { // Testing (8)
    using Tup = cuda::std::tuple<void*, unsigned, char>;
    cuda::std::tuple t1(Tup(nullptr, 42, 'a'));
    ASSERT_SAME_TYPE(decltype(t1), Tup);
    unused(t1);
  }
  // cuda::std::allocator not supported
  /*
  { // Testing (9)
    using Tup = cuda::std::tuple<int, decltype(nullptr)>;
    const Tup t(42, nullptr);

    cuda::std::tuple t1(AT, A, t);
    ASSERT_SAME_TYPE(decltype(t1), Tup);
    unused(t1);
  }
  { // Testing (10)
    using Tup = cuda::std::tuple<void*, unsigned, char>;
    cuda::std::tuple t1(AT, A, Tup(nullptr, 42, 'a'));
    ASSERT_SAME_TYPE(decltype(t1), Tup);
    unused(t1);
  }
  */
}

// Overloads
//  using A = Allocator
//  using AT = cuda::std::allocator_arg_t
// ---------------
// (1)  tuple() -> tuple<>
// (2)  tuple(AT, A const&) -> tuple<>
// (3)  tuple(tuple const&) -> tuple<>
// (4)  tuple(tuple&&) -> tuple<>
// (5)  tuple(AT, A const&, tuple const&) -> tuple<>
// (6)  tuple(AT, A const&, tuple&&) -> tuple<>
__host__ __device__ void test_empty_specialization()
{
  // cuda::std::allocator not supported
  // cuda::std::allocator<int> A;
  const auto AT = cuda::std::allocator_arg;
  unused(AT);
  { // Testing (1)
    cuda::std::tuple t1{};
    ASSERT_SAME_TYPE(decltype(t1), cuda::std::tuple<>);
    unused(t1);
  }
  // cuda::std::allocator not supported
  /*
  { // Testing (2)
    cuda::std::tuple t1{AT, A};
    ASSERT_SAME_TYPE(decltype(t1), cuda::std::tuple<>);
  }
  */
  { // Testing (3)
    const cuda::std::tuple<> t{};
    cuda::std::tuple t1(t);
    ASSERT_SAME_TYPE(decltype(t1), cuda::std::tuple<>);
    unused(t1);
  }
  { // Testing (4)
    cuda::std::tuple t1(cuda::std::tuple<>{});
    ASSERT_SAME_TYPE(decltype(t1), cuda::std::tuple<>);
    unused(t1);
  }
  // cuda::std::allocator not supported
  /*
  { // Testing (5)
    const cuda::std::tuple<> t{};
    cuda::std::tuple t1(AT, A, t);
    ASSERT_SAME_TYPE(decltype(t1), cuda::std::tuple<>);
  }
  { // Testing (6)
    cuda::std::tuple t1(AT, A, cuda::std::tuple<>{});
    ASSERT_SAME_TYPE(decltype(t1), cuda::std::tuple<>);
  }
  */
}

int main(int, char**)
{
  test_primary_template();
  test_empty_specialization();

  return 0;
}
