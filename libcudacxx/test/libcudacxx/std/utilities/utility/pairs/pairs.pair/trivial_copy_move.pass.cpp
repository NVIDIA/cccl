//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// pair(pair const&) = default;
// pair(pair&&) = default;

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct Dummy
{
  Dummy(Dummy const&) = delete;
  Dummy(Dummy&&)      = default;
};

struct NonTrivialMutableCopy
{
  NonTrivialMutableCopy(const NonTrivialMutableCopy&) = default;
  TEST_FUNC NonTrivialMutableCopy(NonTrivialMutableCopy&) {}
};

struct NonTrivialConstMove
{
  NonTrivialConstMove(const NonTrivialConstMove&) = default;
  TEST_FUNC NonTrivialConstMove(const NonTrivialConstMove&&) {}
};

int main(int, char**)
{
  using P = cuda::std::pair<int, short>;
  {
    static_assert(cuda::std::is_copy_constructible<P>::value);
    static_assert(cuda::std::is_trivially_copy_constructible<P>::value);
  }
  {
    static_assert(cuda::std::is_move_constructible<P>::value);
    static_assert(cuda::std::is_trivially_move_constructible<P>::value);
  }
  // pair(pair<U1, U2>&) and pair(const pair<U1, U2>&&) must not replace a trivial copy
  {
    static_assert(cuda::std::is_trivially_constructible_v<P, P&>);
    static_assert(cuda::std::is_trivially_constructible_v<P, const P&&>);
  }
  {
    using P1 = cuda::std::pair<int&, int>;
    static_assert(cuda::std::is_trivially_copy_constructible_v<P1>);
    static_assert(cuda::std::is_trivially_constructible_v<P1, P1&>);
    static_assert(cuda::std::is_trivially_constructible_v<P1, const P1&&>);
  }
  {
    using P1 = cuda::std::pair<int, NonTrivialMutableCopy>;
    static_assert(cuda::std::is_constructible_v<P1, P1&>);
    static_assert(!cuda::std::is_trivially_constructible_v<P1, P1&>);
  }
  {
    using P1 = cuda::std::pair<int, NonTrivialConstMove>;
    static_assert(cuda::std::is_constructible_v<P1, const P1&&>);
    static_assert(!cuda::std::is_trivially_constructible_v<P1, const P1&&>);
  }
  // The copy constructor is deleted for an rvalue reference member, so a const rvalue needs pair(const pair<U1, U2>&&)
  {
    using P1 = cuda::std::pair<int&&, int>;
    static_assert(!cuda::std::is_copy_constructible_v<P1>);
    static_assert(cuda::std::is_constructible_v<P1, const P1&&>);
  }
  {
    using P1 = cuda::std::pair<Dummy, int>;
    static_assert(!cuda::std::is_copy_constructible<P1>::value);
    static_assert(!cuda::std::is_trivially_copy_constructible<P1>::value);
    static_assert(cuda::std::is_move_constructible<P1>::value);
    static_assert(cuda::std::is_trivially_move_constructible<P1>::value);
  }

  // extensions to ensure pair is trivially_copyable
  {
    static_assert(cuda::std::is_copy_assignable<P>::value);
    static_assert(cuda::std::is_trivially_copy_assignable<P>::value);
  }
  {
    static_assert(cuda::std::is_move_assignable<P>::value);
    static_assert(cuda::std::is_trivially_move_assignable<P>::value);
  }
  {
    static_assert(cuda::std::is_trivially_copyable<P>::value);
  }

  return 0;
}
