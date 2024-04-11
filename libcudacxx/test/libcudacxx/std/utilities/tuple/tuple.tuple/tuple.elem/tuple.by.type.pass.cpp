//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

#include <cuda/std/tuple>
#include <cuda/std/utility>
// cuda::std::unique_ptr not supported
// #include <cuda/std/memory>
// cuda::std::string not supported
// #include <cuda/std/string>
// cuda::std::complex not supported
// #include <cuda/std/complex>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  // cuda::std::complex not supported
  // cuda::std::string not supported
  /*
  typedef cuda::std::complex<float> cf;
  {
  auto t1 = cuda::std::tuple<int, cuda::std::string, cf> { 42, "Hi", { 1,2 }};
  assert ( cuda::std::get<int>(t1) == 42 ); // find at the beginning
  assert ( cuda::std::get<cuda::std::string>(t1) == "Hi" ); // find in the middle
  assert ( cuda::std::get<cf>(t1).real() == 1 ); // find at the end
  assert ( cuda::std::get<cf>(t1).imag() == 2 );
  }

  {
  auto t2 = cuda::std::tuple<int, cuda::std::string, int, cf> { 42, "Hi", 23, { 1,2 }};
//  get<int> would fail!
  assert ( cuda::std::get<cuda::std::string>(t2) == "Hi" );
  assert (( cuda::std::get<cf>(t2) == cf{ 1,2 } ));
  }
  */
  {
    constexpr cuda::std::tuple<int, const int, double, double> p5{1, 2, 3.4, 5.6};
    static_assert(cuda::std::get<int>(p5) == 1, "");
    static_assert(cuda::std::get<const int>(p5) == 2, "");
  }

  {
    const cuda::std::tuple<int, const int, double, double> p5{1, 2, 3.4, 5.6};
    const int& i1 = cuda::std::get<int>(p5);
    const int& i2 = cuda::std::get<const int>(p5);
    assert(i1 == 1);
    assert(i2 == 2);
  }

  // cuda::std::unique_ptr not supported
  /*
  {
  typedef cuda::std::unique_ptr<int> upint;
  cuda::std::tuple<upint> t(upint(new int(4)));
  upint p = cuda::std::get<upint>(cuda::std::move(t)); // get rvalue
  assert(*p == 4);
  assert(cuda::std::get<upint>(t) == nullptr); // has been moved from
  }

  {
  typedef cuda::std::unique_ptr<int> upint;
  const cuda::std::tuple<upint> t(upint(new int(4)));
  const upint&& p = cuda::std::get<upint>(cuda::std::move(t)); // get const rvalue
  assert(*p == 4);
  assert(cuda::std::get<upint>(t) != nullptr);
  }
  */

  {
    int x = 42;
    int y = 43;
    cuda::std::tuple<int&, int const&> const t(x, y);
    static_assert(cuda::std::is_same<int&, decltype(cuda::std::get<int&>(cuda::std::move(t)))>::value, "");
    static_assert(noexcept(cuda::std::get<int&>(cuda::std::move(t))), "");
    static_assert(cuda::std::is_same<int const&, decltype(cuda::std::get<int const&>(cuda::std::move(t)))>::value, "");
    static_assert(noexcept(cuda::std::get<int const&>(cuda::std::move(t))), "");
  }

  {
    int x = 42;
    int y = 43;
    cuda::std::tuple<int&&, int const&&> const t(cuda::std::move(x), cuda::std::move(y));
    static_assert(cuda::std::is_same<int&&, decltype(cuda::std::get<int&&>(cuda::std::move(t)))>::value, "");
    static_assert(noexcept(cuda::std::get<int&&>(cuda::std::move(t))), "");
    static_assert(cuda::std::is_same<int const&&, decltype(cuda::std::get<int const&&>(cuda::std::move(t)))>::value,
                  "");
    static_assert(noexcept(cuda::std::get<int const&&>(cuda::std::move(t))), "");
  }
  {
    constexpr const cuda::std::tuple<int, const int, double, double> t{1, 2, 3.4, 5.6};
    static_assert(cuda::std::get<int>(cuda::std::move(t)) == 1, "");
    static_assert(cuda::std::get<const int>(cuda::std::move(t)) == 2, "");
  }

  return 0;
}
