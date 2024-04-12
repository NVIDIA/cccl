//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// tuple(tuple&& u);

// UNSUPPORTED: c++98, c++03

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

struct NothrowConstruct
{
  __host__ __device__ constexpr NothrowConstruct(int) noexcept {};
};

int main(int, char**)
{
  {
    typedef cuda::std::tuple<NothrowConstruct, NothrowConstruct> T;
    T t(0, 1);
    unused(t); // Prevent unused warning

    // Test that tuple<> handles noexcept properly
    static_assert(cuda::std::is_nothrow_constructible<T, int, int>(), "");
    static_assert(cuda::std::is_nothrow_constructible<NothrowConstruct, int>(), "");
  }

  return 0;
}
