//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// UNSUPPORTED: msvc

#include <cuda/std/utility>
// cuda::std::string not supported
// #include <cuda/std/string>
#include <cuda/std/type_traits>
// cuda/std/complex not supported
// #include <cuda/std/complex>
// cuda/std/memory not supported
// #include <cuda/std/memory>

#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  // cuda/std/complex not supported
  /*
  typedef cuda::std::complex<float> cf;
  {
  auto t1 = cuda::std::make_pair<int, cf> ( 42, { 1,2 } );
  assert ( cuda::std::get<int>(t1) == 42 );
  assert ( cuda::std::get<cf>(t1).real() == 1 );
  assert ( cuda::std::get<cf>(t1).imag() == 2 );
  }
  */
  {
    const cuda::std::pair<int, const int> p1{1, 2};
    const int& i1 = cuda::std::get<int>(p1);
    const int& i2 = cuda::std::get<const int>(p1);
    assert(i1 == 1);
    assert(i2 == 2);
  }

  // cuda/std/memory not supported
  /*
  {
  typedef cuda::std::unique_ptr<int> upint;
  cuda::std::pair<upint, int> t(upint(new int(4)), 42);
  upint p = cuda::std::get<upint>(cuda::std::move(t)); // get rvalue
  assert(*p == 4);
  assert(cuda::std::get<upint>(t) == nullptr); // has been moved from
  }

  {
  typedef cuda::std::unique_ptr<int> upint;
  const cuda::std::pair<upint, int> t(upint(new int(4)), 42);
  static_assert(cuda::std::is_same<const upint&&, decltype(cuda::std::get<upint>(cuda::std::move(t)))>::value, "");
  static_assert(noexcept(cuda::std::get<upint>(cuda::std::move(t))), "");
  static_assert(cuda::std::is_same<const int&&, decltype(cuda::std::get<int>(cuda::std::move(t)))>::value, "");
  static_assert(noexcept(cuda::std::get<int>(cuda::std::move(t))), "");
  auto&& p = cuda::std::get<upint>(cuda::std::move(t)); // get const rvalue
  auto&& i = cuda::std::get<int>(cuda::std::move(t)); // get const rvalue
  assert(*p == 4);
  assert(i == 42);
  assert(cuda::std::get<upint>(t) != nullptr);
  }
  */

  {
    int x       = 42;
    int const y = 43;
    cuda::std::pair<int&, int const&> const p(x, y);
    static_assert(cuda::std::is_same<int&, decltype(cuda::std::get<int&>(cuda::std::move(p)))>::value, "");
    static_assert(noexcept(cuda::std::get<int&>(cuda::std::move(p))), "");
    static_assert(cuda::std::is_same<int const&, decltype(cuda::std::get<int const&>(cuda::std::move(p)))>::value, "");
    static_assert(noexcept(cuda::std::get<int const&>(cuda::std::move(p))), "");
  }

  {
    int x       = 42;
    int const y = 43;
    cuda::std::pair<int&&, int const&&> const p(cuda::std::move(x), cuda::std::move(y));
    static_assert(cuda::std::is_same<int&&, decltype(cuda::std::get<int&&>(cuda::std::move(p)))>::value, "");
    static_assert(noexcept(cuda::std::get<int&&>(cuda::std::move(p))), "");
    static_assert(cuda::std::is_same<int const&&, decltype(cuda::std::get<int const&&>(cuda::std::move(p)))>::value,
                  "");
    static_assert(noexcept(cuda::std::get<int const&&>(cuda::std::move(p))), "");
  }

  {
    constexpr const cuda::std::pair<int, const int> p{1, 2};
    static_assert(cuda::std::get<int>(cuda::std::move(p)) == 1, "");
    static_assert(cuda::std::get<const int>(cuda::std::move(p)) == 2, "");
  }

  return 0;
}
