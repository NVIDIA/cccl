//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc, class... UTypes>
//   tuple(allocator_arg_t, const Alloc& a, UTypes&&...);

// UNSUPPORTED: c++98, c++03

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "../alloc_first.h"
#include "../alloc_last.h"
#include "allocators.h"
#include "MoveOnly.h"
#include "test_macros.h"

template <class T = void>
struct DefaultCtorBlowsUp
{
  __host__ __device__ constexpr DefaultCtorBlowsUp()
  {
    static_assert(!cuda::std::is_same<T, T>::value, "Default Ctor instantiated");
  }

  __host__ __device__ explicit constexpr DefaultCtorBlowsUp(int x)
      : value(x)
  {}

  int value;
};

struct DerivedFromAllocArgT : cuda::std::allocator_arg_t
{};

// Make sure the _Up... constructor SFINAEs out when the number of initializers
// is less that the number of elements in the tuple. Previously libc++ would
// offer these constructers as an extension but they broke conforming code.
__host__ __device__ void test_uses_allocator_sfinae_evaluation()
{
  using BadDefault = DefaultCtorBlowsUp<>;
  {
    typedef cuda::std::tuple<MoveOnly, MoveOnly, BadDefault> Tuple;

    static_assert(!cuda::std::is_constructible<Tuple, cuda::std::allocator_arg_t, A1<int>, MoveOnly>::value, "");

    static_assert(
      cuda::std::is_constructible<Tuple, cuda::std::allocator_arg_t, A1<int>, MoveOnly, MoveOnly, BadDefault>::value,
      "");
  }
  {
    typedef cuda::std::tuple<MoveOnly, MoveOnly, BadDefault, BadDefault> Tuple;

    static_assert(!cuda::std::is_constructible<Tuple, cuda::std::allocator_arg_t, A1<int>, MoveOnly, MoveOnly>::value,
                  "");

    static_assert(
      cuda::std::
        is_constructible<Tuple, cuda::std::allocator_arg_t, A1<int>, MoveOnly, MoveOnly, BadDefault, BadDefault>::value,
      "");
  }
}

struct Explicit
{
  int value;
  __host__ __device__ explicit Explicit(int x)
      : value(x)
  {}
};

int main(int, char**)
{
  // cuda::std::allocator not supported
  /*
  {
      cuda::std::tuple<Explicit> t{cuda::std::allocator_arg, cuda::std::allocator<void>{}, 42};
      assert(cuda::std::get<0>(t).value == 42);
  }
  */
  {
    cuda::std::tuple<MoveOnly> t(cuda::std::allocator_arg, A1<int>(), MoveOnly(0));
    assert(cuda::std::get<0>(t) == 0);
  }
  {
    using T = DefaultCtorBlowsUp<>;
    cuda::std::tuple<T> t(cuda::std::allocator_arg, A1<int>(), T(42));
    assert(cuda::std::get<0>(t).value == 42);
  }
  {
    cuda::std::tuple<MoveOnly, MoveOnly> t(cuda::std::allocator_arg, A1<int>(), MoveOnly(0), MoveOnly(1));
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t) == 1);
  }
  {
    using T = DefaultCtorBlowsUp<>;
    cuda::std::tuple<T, T> t(cuda::std::allocator_arg, A1<int>(), T(42), T(43));
    assert(cuda::std::get<0>(t).value == 42);
    assert(cuda::std::get<1>(t).value == 43);
  }
  {
    cuda::std::tuple<MoveOnly, MoveOnly, MoveOnly> t(cuda::std::allocator_arg, A1<int>(), MoveOnly(0), 1, 2);
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t) == 1);
    assert(cuda::std::get<2>(t) == 2);
  }
  {
    using T = DefaultCtorBlowsUp<>;
    cuda::std::tuple<T, T, T> t(cuda::std::allocator_arg, A1<int>(), T(1), T(2), T(3));
    assert(cuda::std::get<0>(t).value == 1);
    assert(cuda::std::get<1>(t).value == 2);
    assert(cuda::std::get<2>(t).value == 3);
  }
  {
    alloc_first::allocator_constructed() = false;
    alloc_last::allocator_constructed()  = false;
    cuda::std::tuple<int, alloc_first, alloc_last> t(cuda::std::allocator_arg, A1<int>(5), 1, 2, 3);
    assert(cuda::std::get<0>(t) == 1);
    assert(alloc_first::allocator_constructed());
    assert(cuda::std::get<1>(t) == alloc_first(2));
    assert(alloc_last::allocator_constructed());
    assert(cuda::std::get<2>(t) == alloc_last(3));
  }
  {
    // Check that uses-allocator construction is still selected when
    // given a tag type that derives from allocator_arg_t.
    DerivedFromAllocArgT tag;
    alloc_first::allocator_constructed() = false;
    alloc_last::allocator_constructed()  = false;
    cuda::std::tuple<int, alloc_first, alloc_last> t(tag, A1<int>(5), 1, 2, 3);
    assert(cuda::std::get<0>(t) == 1);
    assert(alloc_first::allocator_constructed());
    assert(cuda::std::get<1>(t) == alloc_first(2));
    assert(alloc_last::allocator_constructed());
    assert(cuda::std::get<2>(t) == alloc_last(3));
  }
  // Stress test the SFINAE on the uses-allocator constructors and
  // ensure that the "reduced-arity-initialization" extension is not offered
  // for these constructors.
  test_uses_allocator_sfinae_evaluation();

  return 0;
}
