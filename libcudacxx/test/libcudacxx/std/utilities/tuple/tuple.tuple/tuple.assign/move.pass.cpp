//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// tuple& operator=(tuple&& u);

// UNSUPPORTED: c++98, c++03

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

struct NonAssignable
{
  NonAssignable& operator=(NonAssignable const&) = delete;
  NonAssignable& operator=(NonAssignable&&)      = delete;
};
struct CopyAssignable
{
  CopyAssignable& operator=(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable&&)      = delete;
};
static_assert(cuda::std::is_copy_assignable<CopyAssignable>::value, "");
struct MoveAssignable
{
  MoveAssignable& operator=(MoveAssignable const&) = delete;
  MoveAssignable& operator=(MoveAssignable&&)      = default;
};

STATIC_TEST_GLOBAL_VAR int copied = 0;
STATIC_TEST_GLOBAL_VAR int moved  = 0;

struct CountAssign
{
  __host__ __device__ static void reset()
  {
    copied = moved = 0;
  }
  CountAssign() = default;
  __host__ __device__ CountAssign& operator=(CountAssign const&)
  {
    ++copied;
    return *this;
  }
  __host__ __device__ CountAssign& operator=(CountAssign&&)
  {
    ++moved;
    return *this;
  }
};

int main(int, char**)
{
  {
    typedef cuda::std::tuple<> T;
    T t0;
    T t;
    t = cuda::std::move(t0);
    unused(t);
  }
  {
    typedef cuda::std::tuple<MoveOnly> T;
    T t0(MoveOnly(0));
    T t;
    t = cuda::std::move(t0);
    assert(cuda::std::get<0>(t) == 0);
  }
  {
    typedef cuda::std::tuple<MoveOnly, MoveOnly> T;
    T t0(MoveOnly(0), MoveOnly(1));
    T t;
    t = cuda::std::move(t0);
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t) == 1);
  }
  {
    typedef cuda::std::tuple<MoveOnly, MoveOnly, MoveOnly> T;
    T t0(MoveOnly(0), MoveOnly(1), MoveOnly(2));
    T t;
    t = cuda::std::move(t0);
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t) == 1);
    assert(cuda::std::get<2>(t) == 2);
  }
  {
    // test reference assignment.
    using T = cuda::std::tuple<int&, int&&>;
    int x   = 42;
    int y   = 100;
    int x2  = -1;
    int y2  = 500;
    T t(x, cuda::std::move(y));
    T t2(x2, cuda::std::move(y2));
    t = cuda::std::move(t2);
    assert(cuda::std::get<0>(t) == x2);
    assert(&cuda::std::get<0>(t) == &x);
    assert(cuda::std::get<1>(t) == y2);
    assert(&cuda::std::get<1>(t) == &y);
  }
  // cuda::std::unique_ptr not supported
  /*
  {
      // test that the implicitly generated move assignment operator
      // is properly deleted
      using T = cuda::std::tuple<cuda::std::unique_ptr<int>>;
      static_assert(cuda::std::is_move_assignable<T>::value, "");
      static_assert(!cuda::std::is_copy_assignable<T>::value, "");
  }
  */
  {
    using T = cuda::std::tuple<int, NonAssignable>;
    static_assert(!cuda::std::is_move_assignable<T>::value, "");
  }
  {
    using T = cuda::std::tuple<int, MoveAssignable>;
    static_assert(cuda::std::is_move_assignable<T>::value, "");
  }
  {
    // The move should decay to a copy.
    CountAssign::reset();
    using T = cuda::std::tuple<CountAssign, CopyAssignable>;
    static_assert(cuda::std::is_move_assignable<T>::value, "");
    T t1;
    T t2;
    t1 = cuda::std::move(t2);
    assert(copied == 1);
    assert(moved == 0);
  }

  return 0;
}
