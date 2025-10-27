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
//   struct tuple_size<tuple<Types...>>
//     : public integral_constant<size_t, sizeof...(Types)> { };

// XFAIL: gcc-4.8, gcc-4.9

#include <cuda/std/tuple>
// cuda::std::array not supported
// #include <cuda/std/array>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, size_t Size = sizeof(cuda::std::tuple_size<T>)>
__host__ __device__ constexpr bool is_complete(int)
{
  static_assert(Size > 0, "");
  return true;
}
template <class>
__host__ __device__ constexpr bool is_complete(long)
{
  return false;
}
template <class T>
__host__ __device__ constexpr bool is_complete()
{
  return is_complete<T>(0);
}

struct Dummy1
{};
struct Dummy2
{};

namespace cuda::std
{
template <>
struct tuple_size<Dummy1> : public integral_constant<size_t, 0>
{};
} // namespace cuda::std

template <class T>
__host__ __device__ void test_complete()
{
  static_assert(is_complete<T>(), "");
  static_assert(is_complete<const T>(), "");
  static_assert(is_complete<volatile T>(), "");
  static_assert(is_complete<const volatile T>(), "");
}

template <class T>
__host__ __device__ void test_incomplete()
{
  static_assert(!is_complete<T>(), "");
  static_assert(!is_complete<const T>(), "");
  static_assert(!is_complete<volatile T>(), "");
  static_assert(!is_complete<const volatile T>(), "");
}

int main(int, char**)
{
  test_complete<cuda::std::tuple<>>();
  test_complete<cuda::std::tuple<int&>>();
  test_complete<cuda::std::tuple<int&&, int&, void*>>();
  test_complete<cuda::std::pair<int, long>>();
  // cuda::std::array not supported
  // test_complete<cuda::std::array<int, 5> >();
  test_complete<Dummy1>();

  test_incomplete<void>();
  test_incomplete<int>();
  test_incomplete<cuda::std::tuple<int>&>();
  test_incomplete<Dummy2>();

  return 0;
}
