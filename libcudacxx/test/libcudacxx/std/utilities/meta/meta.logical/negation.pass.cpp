//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// type_traits

// template<class B> struct negation;                        // C++17
// template<class B>
//   constexpr bool negation_v = negation<B>::value;         // C++17

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct True
{
  static constexpr bool value = true;
};
struct False
{
  static constexpr bool value = false;
};

int main(int, char**)
{
  static_assert(!cuda::std::negation<cuda::std::true_type>::value, "");
  static_assert(cuda::std::negation<cuda::std::false_type>::value, "");

  static_assert(!cuda::std::negation_v<cuda::std::true_type>, "");
  static_assert(cuda::std::negation_v<cuda::std::false_type>, "");

  static_assert(!cuda::std::negation<True>::value, "");
  static_assert(cuda::std::negation<False>::value, "");

  static_assert(!cuda::std::negation_v<True>, "");
  static_assert(cuda::std::negation_v<False>, "");

  static_assert(cuda::std::negation<cuda::std::negation<cuda::std::true_type>>::value, "");
  static_assert(!cuda::std::negation<cuda::std::negation<cuda::std::false_type>>::value, "");

  return 0;
}
