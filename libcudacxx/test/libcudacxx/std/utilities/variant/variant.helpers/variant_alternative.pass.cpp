//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>

// template <size_t I, class T> struct variant_alternative; // undefined
// template <size_t I, class T> struct variant_alternative<I, const T>;
// template <size_t I, class T> struct variant_alternative<I, volatile T>;
// template <size_t I, class T> struct variant_alternative<I, const volatile T>;
// template <size_t I, class T>
//   using variant_alternative_t = typename variant_alternative<I, T>::type;
//
// template <size_t I, class... Types>
//    struct variant_alternative<I, variant<Types...>>;

// #include <cuda/std/memory>
#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

template <class V, size_t I, class E>
__host__ __device__ void test()
{
  static_assert(cuda::std::is_same_v<typename cuda::std::variant_alternative<I, V>::type, E>, "");
  static_assert(cuda::std::is_same_v<typename cuda::std::variant_alternative<I, const V>::type, const E>, "");
  static_assert(cuda::std::is_same_v<typename cuda::std::variant_alternative<I, volatile V>::type, volatile E>, "");
  static_assert(
    cuda::std::is_same_v<typename cuda::std::variant_alternative<I, const volatile V>::type, const volatile E>, "");
  static_assert(cuda::std::is_same_v<cuda::std::variant_alternative_t<I, V>, E>, "");
  static_assert(cuda::std::is_same_v<cuda::std::variant_alternative_t<I, const V>, const E>, "");
  static_assert(cuda::std::is_same_v<cuda::std::variant_alternative_t<I, volatile V>, volatile E>, "");
  static_assert(cuda::std::is_same_v<cuda::std::variant_alternative_t<I, const volatile V>, const volatile E>, "");
}

int main(int, char**)
{
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
  {
    using V = cuda::std::variant<int, void*, const void*, long double>;
    test<V, 0, int>();
    test<V, 1, void*>();
    test<V, 2, const void*>();

    test<V, 3, long double>();
  }
#endif // _LIBCUDACXX_HAS_NO_LONG_DOUBLE
#if !defined(TEST_VARIANT_HAS_NO_REFERENCES)
  {
    using V = cuda::std::variant<int, int&, const int&, int&&, long double>;
    test<V, 0, int>();
    test<V, 1, int&>();
    test<V, 2, const int&>();
    test<V, 3, int&&>();
    test<V, 4, long double>();
  }
#endif

  return 0;
}
