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

// UNSUPPORTED: c++98, c++03

#include <cuda/std/array>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

struct Dummy1
{};
struct Dummy2
{};
struct Dummy3
{};

template <>
struct cuda::std::tuple_size<Dummy1>
{
public:
  static size_t value;
};

template <>
struct cuda::std::tuple_size<Dummy2>
{
public:
  __host__ __device__ static void value() {}
};

template <>
struct cuda::std::tuple_size<Dummy3>
{};

int main(int, char**)
{
  // Test that tuple_size<const T> is not incomplete when tuple_size<T>::value
  // is well-formed but not a constant expression.
  {
    // expected-error@*:* 1 {{is not a constant expression}}
    (void) cuda::std::tuple_size<const Dummy1>::value; // expected-note {{here}}
  }
  // Test that tuple_size<const T> is not incomplete when tuple_size<T>::value
  // is well-formed but not convertible to size_t.
  {
    // expected-error@*:* 1 {{value of type 'void ()' is not implicitly convertible to}}
    (void) cuda::std::tuple_size<const Dummy2>::value; // expected-note {{here}}
  }
  // Test that tuple_size<const T> generates an error when tuple_size<T> is
  // complete but ::value isn't a constant expression convertible to size_t.
  {
    // expected-error@*:* 1 {{no member named 'value'}}
    (void) cuda::std::tuple_size<const Dummy3>::value; // expected-note {{here}}
  }

  return 0;
}
