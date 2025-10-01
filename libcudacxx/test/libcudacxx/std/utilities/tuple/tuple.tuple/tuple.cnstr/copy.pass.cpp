//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// tuple(const tuple& u) = default;

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

struct Empty
{};

int main(int, char**)
{
  {
    using T = cuda::std::tuple<>;
    T t0;
    T t = t0;
    unused(t); // Prevent unused warning
  }
  {
    using T = cuda::std::tuple<int>;
    T t0(2);
    T t = t0;
    assert(cuda::std::get<0>(t) == 2);
  }
  {
    using T = cuda::std::tuple<int, char>;
    T t0(2, 'a');
    T t = t0;
    assert(cuda::std::get<0>(t) == 2);
    assert(cuda::std::get<1>(t) == 'a');
  }
  // cuda::std::string not supported
  /*
  {
      using T = cuda::std::tuple<int, char, cuda::std::string>;
      const T t0(2, 'a', "some text");
      T t = t0;
      assert(cuda::std::get<0>(t) == 2);
      assert(cuda::std::get<1>(t) == 'a');
      assert(cuda::std::get<2>(t) == "some text");
  }
  */
  {
    using T = cuda::std::tuple<int>;
    constexpr T t0(2);
    constexpr T t = t0;
    static_assert(cuda::std::get<0>(t) == 2, "");
  }
  {
    using T = cuda::std::tuple<Empty>;
    constexpr T t0;
    constexpr T t                      = t0;
    [[maybe_unused]] constexpr Empty e = cuda::std::get<0>(t);
  }

  return 0;
}
