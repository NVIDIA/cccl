//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// type_traits

// template<class... B> struct disjunction;                           // C++17
// template<class... B>
//   constexpr bool disjunction_v = disjunction<B...>::value;         // C++17

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
  static_assert(!cuda::std::disjunction<>::value, "");
  static_assert(cuda::std::disjunction<cuda::std::true_type>::value, "");
  static_assert(!cuda::std::disjunction<cuda::std::false_type>::value, "");

  static_assert(!cuda::std::disjunction_v<>, "");
  static_assert(cuda::std::disjunction_v<cuda::std::true_type>, "");
  static_assert(!cuda::std::disjunction_v<cuda::std::false_type>, "");

  static_assert(cuda::std::disjunction<cuda::std::true_type, cuda::std::true_type>::value, "");
  static_assert(cuda::std::disjunction<cuda::std::true_type, cuda::std::false_type>::value, "");
  static_assert(cuda::std::disjunction<cuda::std::false_type, cuda::std::true_type>::value, "");
  static_assert(!cuda::std::disjunction<cuda::std::false_type, cuda::std::false_type>::value, "");

  static_assert(cuda::std::disjunction_v<cuda::std::true_type, cuda::std::true_type>, "");
  static_assert(cuda::std::disjunction_v<cuda::std::true_type, cuda::std::false_type>, "");
  static_assert(cuda::std::disjunction_v<cuda::std::false_type, cuda::std::true_type>, "");
  static_assert(!cuda::std::disjunction_v<cuda::std::false_type, cuda::std::false_type>, "");

  static_assert(cuda::std::disjunction<cuda::std::true_type, cuda::std::true_type, cuda::std::true_type>::value, "");
  static_assert(cuda::std::disjunction<cuda::std::true_type, cuda::std::false_type, cuda::std::true_type>::value, "");
  static_assert(cuda::std::disjunction<cuda::std::false_type, cuda::std::true_type, cuda::std::true_type>::value, "");
  static_assert(cuda::std::disjunction<cuda::std::false_type, cuda::std::false_type, cuda::std::true_type>::value, "");
  static_assert(cuda::std::disjunction<cuda::std::true_type, cuda::std::true_type, cuda::std::false_type>::value, "");
  static_assert(cuda::std::disjunction<cuda::std::true_type, cuda::std::false_type, cuda::std::false_type>::value, "");
  static_assert(cuda::std::disjunction<cuda::std::false_type, cuda::std::true_type, cuda::std::false_type>::value, "");
  static_assert(!cuda::std::disjunction<cuda::std::false_type, cuda::std::false_type, cuda::std::false_type>::value,
                "");

  static_assert(cuda::std::disjunction_v<cuda::std::true_type, cuda::std::true_type, cuda::std::true_type>, "");
  static_assert(cuda::std::disjunction_v<cuda::std::true_type, cuda::std::false_type, cuda::std::true_type>, "");
  static_assert(cuda::std::disjunction_v<cuda::std::false_type, cuda::std::true_type, cuda::std::true_type>, "");
  static_assert(cuda::std::disjunction_v<cuda::std::false_type, cuda::std::false_type, cuda::std::true_type>, "");
  static_assert(cuda::std::disjunction_v<cuda::std::true_type, cuda::std::true_type, cuda::std::false_type>, "");
  static_assert(cuda::std::disjunction_v<cuda::std::true_type, cuda::std::false_type, cuda::std::false_type>, "");
  static_assert(cuda::std::disjunction_v<cuda::std::false_type, cuda::std::true_type, cuda::std::false_type>, "");
  static_assert(!cuda::std::disjunction_v<cuda::std::false_type, cuda::std::false_type, cuda::std::false_type>, "");

  static_assert(cuda::std::disjunction<True>::value, "");
  static_assert(!cuda::std::disjunction<False>::value, "");

  static_assert(cuda::std::disjunction_v<True>, "");
  static_assert(!cuda::std::disjunction_v<False>, "");

  return 0;
}
