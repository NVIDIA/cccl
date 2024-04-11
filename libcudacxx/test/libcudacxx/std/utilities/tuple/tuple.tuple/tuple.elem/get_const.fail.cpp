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

// UNSUPPORTED: c++98, c++03

#include <cuda/std/tuple>
// cuda::std::string not supported
// #include <cuda/std/string>
#include <cuda/std/cassert>

int main(int, char**)
{
  // cuda::std::string not supported
  /*
  {
      typedef cuda::std::tuple<double&, cuda::std::string, int> T;
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

      cuda::std::get<1>(t) = "four";
  }
  */
  {
    typedef cuda::std::tuple<double&, int> T;
    double d = 1.5;
    const T t(d, 5);
    assert(cuda::std::get<0>(t) == 1.5);
    assert(cuda::std::get<1>(t) == 5);
    cuda::std::get<0>(t) = 2.5;
    assert(cuda::std::get<0>(t) == 2.5);
    assert(cuda::std::get<1>(t) == 5);
    assert(d == 2.5);

    // Expected failure: <1> is not a modifiable lvalue
    cuda::std::get<1>(t) = 10;
  }
  return 0;
}
