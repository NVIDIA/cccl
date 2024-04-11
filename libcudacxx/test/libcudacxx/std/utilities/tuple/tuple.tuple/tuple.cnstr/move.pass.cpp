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

struct ConstructsWithTupleLeaf
{
  __host__ __device__ ConstructsWithTupleLeaf() {}

  __host__ __device__ ConstructsWithTupleLeaf(ConstructsWithTupleLeaf const&)
  {
    assert(false);
  }
  __host__ __device__ ConstructsWithTupleLeaf(ConstructsWithTupleLeaf&&) {}

  template <class T>
  __host__ __device__ ConstructsWithTupleLeaf(T)
  {
    static_assert(!cuda::std::is_same<T, T>::value, "Constructor instantiated for type other than int");
  }
};

// move_only type which triggers the empty base optimization
struct move_only_ebo
{
  move_only_ebo()                = default;
  move_only_ebo(move_only_ebo&&) = default;
};

// a move_only type which does not trigger the empty base optimization
struct move_only_large final
{
  __host__ __device__ move_only_large()
      : value(42)
  {}
  move_only_large(move_only_large&&) = default;
  int value;
};

template <class Elem>
__host__ __device__ void test_sfinae()
{
  using Tup = cuda::std::tuple<Elem>;
  // cuda::std::allocator not supported
  // using Alloc = cuda::std::allocator<void>;
  // using Tag = cuda::std::allocator_arg_t;
  // special members
  {
    static_assert(cuda::std::is_default_constructible<Tup>::value, "");
    static_assert(cuda::std::is_move_constructible<Tup>::value, "");
    static_assert(!cuda::std::is_copy_constructible<Tup>::value, "");
    static_assert(!cuda::std::is_constructible<Tup, Tup&>::value, "");
  }
  // args constructors
  {
    static_assert(cuda::std::is_constructible<Tup, Elem&&>::value, "");
    static_assert(!cuda::std::is_constructible<Tup, Elem const&>::value, "");
    static_assert(!cuda::std::is_constructible<Tup, Elem&>::value, "");
  }
  // cuda::std::allocator not supported
  /*
  // uses-allocator special member constructors
  {
      static_assert(cuda::std::is_constructible<Tup, Tag, Alloc>::value, "");
      static_assert(cuda::std::is_constructible<Tup, Tag, Alloc, Tup&&>::value, "");
      static_assert(!cuda::std::is_constructible<Tup, Tag, Alloc, Tup const&>::value, "");
      static_assert(!cuda::std::is_constructible<Tup, Tag, Alloc, Tup &>::value, "");
  }
  // uses-allocator args constructors
  {
      static_assert(cuda::std::is_constructible<Tup, Tag, Alloc, Elem&&>::value, "");
      static_assert(!cuda::std::is_constructible<Tup, Tag, Alloc, Elem const&>::value, "");
      static_assert(!cuda::std::is_constructible<Tup, Tag, Alloc, Elem &>::value, "");
  }
  */
}

int main(int, char**)
{
  {
    typedef cuda::std::tuple<> T;
    T t0;
    T t = cuda::std::move(t0);
    unused(t); // Prevent unused warning
  }
  {
    typedef cuda::std::tuple<MoveOnly> T;
    T t0(MoveOnly(0));
    T t = cuda::std::move(t0);
    assert(cuda::std::get<0>(t) == 0);
  }
  {
    typedef cuda::std::tuple<MoveOnly, MoveOnly> T;
    T t0(MoveOnly(0), MoveOnly(1));
    T t = cuda::std::move(t0);
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t) == 1);
  }
  {
    typedef cuda::std::tuple<MoveOnly, MoveOnly, MoveOnly> T;
    T t0(MoveOnly(0), MoveOnly(1), MoveOnly(2));
    T t = cuda::std::move(t0);
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t) == 1);
    assert(cuda::std::get<2>(t) == 2);
  }
  // A bug in tuple caused __tuple_leaf to use its explicit converting constructor
  //  as its move constructor. This tests that ConstructsWithTupleLeaf is not called
  // (w/ __tuple_leaf)
  {
    typedef cuda::std::tuple<ConstructsWithTupleLeaf> d_t;
    d_t d((ConstructsWithTupleLeaf()));
    d_t d2(static_cast<d_t&&>(d));
    unused(d2);
  }
  {
    test_sfinae<move_only_ebo>();
    test_sfinae<move_only_large>();
  }

  return 0;
}
