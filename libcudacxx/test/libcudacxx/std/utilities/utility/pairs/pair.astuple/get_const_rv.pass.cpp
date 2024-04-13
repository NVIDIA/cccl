//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// template<size_t I, class T1, class T2>
//     const typename tuple_element<I, cuda::std::pair<T1, T2> >::type&&
//     get(const pair<T1, T2>&&);

// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: msvc

#include <cuda/std/tuple>
#include <cuda/std/utility>
// cuda/std/memory not supported
// #include <cuda/std/memory>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  // cuda/std/memory not supported
  /*
  {
  typedef cuda::std::pair<cuda::std::unique_ptr<int>, short> P;
  const P p(cuda::std::unique_ptr<int>(new int(3)), static_cast<short>(4));
  static_assert(cuda::std::is_same<const cuda::std::unique_ptr<int>&&,
  decltype(cuda::std::get<0>(cuda::std::move(p)))>::value, "");
  static_assert(noexcept(cuda::std::get<0>(cuda::std::move(p))), "");
  const cuda::std::unique_ptr<int>&& ptr = cuda::std::get<0>(cuda::std::move(p));
  assert(*ptr == 3);
  }
  */
  {
    int x       = 42;
    int const y = 43;
    cuda::std::pair<int&, int const&> const p(x, y);
    static_assert(cuda::std::is_same<int&, decltype(cuda::std::get<0>(cuda::std::move(p)))>::value, "");
    static_assert(noexcept(cuda::std::get<0>(cuda::std::move(p))), "");
    static_assert(cuda::std::is_same<int const&, decltype(cuda::std::get<1>(cuda::std::move(p)))>::value, "");
    static_assert(noexcept(cuda::std::get<1>(cuda::std::move(p))), "");
  }

  {
    int x       = 42;
    int const y = 43;
    cuda::std::pair<int&&, int const&&> const p(cuda::std::move(x), cuda::std::move(y));
    static_assert(cuda::std::is_same<int&&, decltype(cuda::std::get<0>(cuda::std::move(p)))>::value, "");
    static_assert(noexcept(cuda::std::get<0>(cuda::std::move(p))), "");
    static_assert(cuda::std::is_same<int const&&, decltype(cuda::std::get<1>(cuda::std::move(p)))>::value, "");
    static_assert(noexcept(cuda::std::get<1>(cuda::std::move(p))), "");
  }

#if TEST_STD_VER > 2011
  {
    typedef cuda::std::pair<int, short> P;
    constexpr const P p1(3, static_cast<short>(4));
    static_assert(cuda::std::get<0>(cuda::std::move(p1)) == 3, "");
    static_assert(cuda::std::get<1>(cuda::std::move(p1)) == 4, "");
  }
#endif

  return 0;
}
