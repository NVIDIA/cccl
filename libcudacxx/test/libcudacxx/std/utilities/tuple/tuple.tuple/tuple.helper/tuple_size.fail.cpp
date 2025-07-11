//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types>
//   class tuple_size<tuple<Types...>>
//     : public integral_constant<size_t, sizeof...(Types)> { };

// Expect failures with a reference type, pointer type, and a non-tuple type.

#include <cuda/std/tuple>

int main(int, char**)
{
  [[maybe_unused]] cuda::std::tuple_size<cuda::std::tuple<>&>::value; // expected-error {{implicit instantiation of
                                                                      // undefined template}}
  [[maybe_unused]] cuda::std::tuple_size<int>::value; // expected-error {{implicit instantiation of undefined template}}
  [[maybe_unused]] cuda::std::tuple_size<cuda::std::tuple<>*>::value; // expected-error {{implicit instantiation of
                                                                      // undefined template}}

  return 0;
}
