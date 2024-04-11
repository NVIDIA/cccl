//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <size_t I, class... Types>
// struct tuple_element<I, tuple<Types...> >
// {
//     typedef Ti type;
// };

// UNSUPPORTED: c++98, c++03

#include <cuda/std/tuple>
#include <cuda/std/type_traits>

int main(int, char**)
{
  using T  = cuda::std::tuple<int, long, void*>;
  using E1 = typename cuda::std::tuple_element<1, T&>::type; // expected-error{{undefined template}}
  using E2 = typename cuda::std::tuple_element<3, T>::type;
  using E3 = typename cuda::std::tuple_element<4, T const>::type;
  // expected-error@*:* 2 {{{{(static_assert|static assertion)}} failed}}

  return 0;
}
