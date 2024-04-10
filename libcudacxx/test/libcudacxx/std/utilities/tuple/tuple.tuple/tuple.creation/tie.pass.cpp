//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template<class... Types>
//   tuple<Types&...> tie(Types&... t);

// UNSUPPORTED: c++98, c++03

#include <cuda/std/tuple>

// cuda::std::string not supported
// #include <cuda/std/string>
#include <cuda/std/cassert>

#include "test_macros.h"

#if TEST_STD_VER > 2011
__host__ __device__ constexpr bool test_tie_constexpr()
{
  {
    int i         = 42;
    double f      = 1.1;
    using ExpectT = cuda::std::tuple<int&, decltype(cuda::std::ignore)&, double&>;
    auto res      = cuda::std::tie(i, cuda::std::ignore, f);
    static_assert(cuda::std::is_same<ExpectT, decltype(res)>::value, "");
    assert(&cuda::std::get<0>(res) == &i);
    assert(&cuda::std::get<1>(res) == &cuda::std::ignore);
    assert(&cuda::std::get<2>(res) == &f);
    // FIXME: If/when tuple gets constexpr assignment
    // res = cuda::std::make_tuple(101, nullptr, -1.0);
  }
  return true;
}
#endif

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, ({
                 int i          = 0;
                 const char* _s = "C++";
                 // cuda::std::string not supported
                 // cuda::std::string s;
                 const char* s;
                 cuda::std::tie(i, cuda::std::ignore, s) = cuda::std::make_tuple(42, 3.14, _s);
                 assert(i == 42);
                 assert(s == _s);
               }))
#if TEST_STD_VER > 2011
  {
    static constexpr int i                                  = 42;
    static constexpr double f                               = 1.1;
    constexpr cuda::std::tuple<const int&, const double&> t = cuda::std::tie(i, f);
    static_assert(cuda::std::get<0>(t) == 42, "");
    static_assert(cuda::std::get<1>(t) == 1.1, "");
  }
  {
    static_assert(test_tie_constexpr(), "");
  }
#endif

  return 0;
}
