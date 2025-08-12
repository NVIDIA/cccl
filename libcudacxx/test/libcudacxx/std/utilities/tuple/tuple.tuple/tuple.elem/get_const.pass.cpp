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
//   typename tuple_element<I, tuple<Types...> >::type const&
//   get(const tuple<Types...>& t);

#include <cuda/std/tuple>
// cuda::std::string not supported
// #include <cuda/std/string>
#include <cuda/std/cassert>

#include "test_macros.h"

struct Empty
{};

int main(int, char**)
{
  {
    using T = cuda::std::tuple<int>;
    const T t(3);
    assert(cuda::std::get<0>(t) == 3);
  }
  // cuda::std::string not supported
  /*
  {
      using T = cuda::std::tuple<cuda::std::string, int>;
      const T t("high", 5);
      assert(cuda::std::get<0>(t) == "high");
      assert(cuda::std::get<1>(t) == 5);
  }
  */
  {
    using T = cuda::std::tuple<double, int>;
    constexpr T t(2.718, 5);
    static_assert(cuda::std::get<0>(t) == 2.718, "");
    static_assert(cuda::std::get<1>(t) == 5, "");
  }
  {
    using T = cuda::std::tuple<Empty>;
    constexpr T t{Empty()};
    [[maybe_unused]] constexpr Empty e = cuda::std::get<0>(t);
  }
  // cuda::std::string not supported
  /*
  {
      using T = cuda::std::tuple<double&, cuda::std::string, int>;
      double d = 1.5;
      const T t(d, "high", 5);
      assert(cuda::std::get<0>(t) == 1.5);
      assert(cuda::std::get<1>(t) == "high");
      assert(cuda::std::get<2>(t) == 5);
      cuda::std::get<0>(t) = 2.5;
      assert(cuda::std::get<0>(t) == 2.5);
      assert(cuda::std::get<1>(t) == "high");
      assert(cuda::std::get<2>(t) == 5);
      assert(d == 2.5);
  }
  */
  {
    using T  = cuda::std::tuple<double&, int>;
    double d = 1.5;
    const T t(d, 5);
    assert(cuda::std::get<0>(t) == 1.5);
    assert(cuda::std::get<1>(t) == 5);
    cuda::std::get<0>(t) = 2.5;
    assert(cuda::std::get<0>(t) == 2.5);
    assert(cuda::std::get<1>(t) == 5);
    assert(d == 2.5);
  }
  return 0;
}
