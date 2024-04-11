//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: msvc

// <utility>

// template <class T1, class T2> struct pair

// pair& operator=(pair const& p);

#include <cuda/std/utility>
// cuda/std/memory not supported
// #include <cuda/std/memory>
#include <cuda/std/cassert>

#include "test_macros.h"

struct NonAssignable
{
  NonAssignable& operator=(NonAssignable const&) = delete;
  NonAssignable& operator=(NonAssignable&&)      = delete;
};
struct CopyAssignable
{
  CopyAssignable()                                 = default;
  CopyAssignable(CopyAssignable const&)            = default;
  CopyAssignable& operator=(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable&&)      = delete;
};
struct MoveAssignable
{
  MoveAssignable()                                 = default;
  MoveAssignable& operator=(MoveAssignable const&) = delete;
  MoveAssignable& operator=(MoveAssignable&&)      = default;
};

struct CountAssign
{
  STATIC_MEMBER_VAR(copied, int);
  STATIC_MEMBER_VAR(moved, int);
  __host__ __device__ static void reset()
  {
    copied() = moved() = 0;
  }
  CountAssign() = default;
  __host__ __device__ CountAssign& operator=(CountAssign const&)
  {
    ++copied();
    return *this;
  }
  __host__ __device__ CountAssign& operator=(CountAssign&&)
  {
    ++moved();
    return *this;
  }
};

struct Incomplete;
TEST_ACCESSIBLE extern Incomplete inc_obj;

int main(int, char**)
{
  {
    typedef cuda::std::pair<CopyAssignable, short> P;
    const P p1(CopyAssignable(), 4);
    P p2;
    p2 = p1;
    assert(p2.second == 4);
  }
  {
    using P = cuda::std::pair<int&, int&&>;
    int x   = 42;
    int y   = 101;
    int x2  = -1;
    int y2  = 300;
    P p1(x, cuda::std::move(y));
    P p2(x2, cuda::std::move(y2));
    p1 = p2;
    assert(p1.first == x2);
    assert(p1.second == y2);
  }
  {
    using P = cuda::std::pair<int, NonAssignable>;
    static_assert(!cuda::std::is_copy_assignable<P>::value, "");
  }
  {
    CountAssign::reset();
    using P = cuda::std::pair<CountAssign, CopyAssignable>;
    static_assert(cuda::std::is_copy_assignable<P>::value, "");
    P p;
    P p2;
    p = p2;
    assert(CountAssign::copied() == 1);
    assert(CountAssign::moved() == 0);
  }
  {
    using P = cuda::std::pair<int, MoveAssignable>;
    static_assert(!cuda::std::is_copy_assignable<P>::value, "");
  }
  {
    using P = cuda::std::pair<int, Incomplete&>;
    static_assert(!cuda::std::is_copy_assignable<P>::value, "");
    P p(42, inc_obj);
    unused(p);
    assert(&p.second == &inc_obj);
  }

  return 0;
}

struct Incomplete
{};
TEST_ACCESSIBLE Incomplete inc_obj;
