//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... Types>
//   class tuple_size<tuple<Types...>>
//     : public integral_constant<size_t, sizeof...(Types)> { };

// XFAIL: gcc-4.8, gcc-4.9

// UNSUPPORTED: c++98, c++03

#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, class = decltype(cuda::std::tuple_size<T>::value)>
__host__ __device__ constexpr bool has_value(int)
{
  return true;
}
template <class>
__host__ __device__ constexpr bool has_value(long)
{
  return false;
}
template <class T>
__host__ __device__ constexpr bool has_value()
{
  return has_value<T>(0);
}

struct Dummy
{};

int main(int, char**)
{
  // Test that the ::value member does not exist
  static_assert(has_value<cuda::std::tuple<int> const>(), "");
  static_assert(has_value<cuda::std::pair<int, long> volatile>(), "");
  static_assert(!has_value<int>(), "");
  static_assert(!has_value<const int>(), "");
  static_assert(!has_value<volatile void>(), "");
  static_assert(!has_value<const volatile cuda::std::tuple<int>&>(), "");

  return 0;
}
