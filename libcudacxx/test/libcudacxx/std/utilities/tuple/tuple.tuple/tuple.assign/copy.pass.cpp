//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// tuple& operator=(const tuple& u);

// UNSUPPORTED: c++98, c++03

#include <cuda/std/cassert>
#include <cuda/std/tuple>

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

int main(int, char**)
{
  {
    typedef cuda::std::tuple<> T;
    T t0;
    T t;
    t = t0;
    unused(t);
  }
  {
    typedef cuda::std::tuple<int> T;
    T t0(2);
    T t;
    t = t0;
    assert(cuda::std::get<0>(t) == 2);
  }
  {
    typedef cuda::std::tuple<int, char> T;
    T t0(2, 'a');
    T t;
    t = t0;
    assert(cuda::std::get<0>(t) == 2);
    assert(cuda::std::get<1>(t) == 'a');
  }
  // cuda::std::string not supported
  /*
  {
      typedef cuda::std::tuple<int, char, cuda::std::string> T;
      const T t0(2, 'a', "some text");
      T t;
      t = t0;
      assert(cuda::std::get<0>(t) == 2);
      assert(cuda::std::get<1>(t) == 'a');
      assert(cuda::std::get<2>(t) == "some text");
  }
  */
  {
    // test reference assignment.
    using T = cuda::std::tuple<int&, int&&>;
    int x   = 42;
    int y   = 100;
    int x2  = -1;
    int y2  = 500;
    T t(x, cuda::std::move(y));
    T t2(x2, cuda::std::move(y2));
    t = t2;
    assert(cuda::std::get<0>(t) == x2);
    assert(&cuda::std::get<0>(t) == &x);
    assert(cuda::std::get<1>(t) == y2);
    assert(&cuda::std::get<1>(t) == &y);
  }
  // cuda::std::unique_ptr not supported
  /*
  {
      // test that the implicitly generated copy assignment operator
      // is properly deleted
      using T = cuda::std::tuple<cuda::std::unique_ptr<int>>;
      static_assert(!cuda::std::is_copy_assignable<T>::value, "");
  }
  */
  {
    using T = cuda::std::tuple<int, NonAssignable>;
    static_assert(!cuda::std::is_copy_assignable<T>::value, "");
  }
  {
    using T = cuda::std::tuple<int, CopyAssignable>;
    static_assert(cuda::std::is_copy_assignable<T>::value, "");
  }
  {
    using T = cuda::std::tuple<int, MoveAssignable>;
    static_assert(!cuda::std::is_copy_assignable<T>::value, "");
  }

  return 0;
}
