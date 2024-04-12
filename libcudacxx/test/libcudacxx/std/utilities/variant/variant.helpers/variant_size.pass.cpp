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

// template <class T> struct variant_size; // undefined
// template <class T> struct variant_size<const T>;
// template <class T> struct variant_size<volatile T>;
// template <class T> struct variant_size<const volatile T>;
// template <class T> constexpr size_t variant_size_v
//     = variant_size<T>::value;

// #include <cuda/std/memory>
#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "test_macros.h"

template <class V, size_t E>
__host__ __device__ void test()
{
  static_assert(cuda::std::variant_size<V>::value == E, "");
  static_assert(cuda::std::variant_size<const V>::value == E, "");
  static_assert(cuda::std::variant_size<volatile V>::value == E, "");
  static_assert(cuda::std::variant_size<const volatile V>::value == E, "");
  static_assert(cuda::std::variant_size_v<V> == E, "");
  static_assert(cuda::std::variant_size_v<const V> == E, "");
  static_assert(cuda::std::variant_size_v<volatile V> == E, "");
  static_assert(cuda::std::variant_size_v<const volatile V> == E, "");
  static_assert(
    cuda::std::is_base_of<cuda::std::integral_constant<cuda::std::size_t, E>, cuda::std::variant_size<V>>::value, "");
};

int main(int, char**)
{
  test<cuda::std::variant<>, 0>();
  test<cuda::std::variant<void*>, 1>();
  test<cuda::std::variant<long, long, void*, double>, 4>();

  return 0;
}
