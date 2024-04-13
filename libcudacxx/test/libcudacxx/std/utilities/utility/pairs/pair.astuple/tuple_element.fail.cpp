//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// tuple_element<I, pair<T1, T2> >::type

#include <cuda/std/utility>

int main(int, char**)
{
  typedef cuda::std::pair<int, short> T;
  cuda::std::tuple_element<2, T>::type foo; // expected-error@__utility/pair.h:* {{Index out of bounds in
                                            // cuda::std::tuple_element<cuda::std::pair<T1, T2>>}}

  return 0;
}
