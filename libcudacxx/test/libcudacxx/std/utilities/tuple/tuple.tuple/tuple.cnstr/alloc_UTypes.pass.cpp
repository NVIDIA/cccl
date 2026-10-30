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

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "../alloc_constexpr_types.h"
#include "allocators.h"
#include "MoveOnly.h"
#include "test_macros.h"

template <class T = void>
struct DefaultCtorBlowsUp
{
  TEST_FUNC constexpr DefaultCtorBlowsUp()
  {
    static_assert(!cuda::std::is_same<T, T>::value, "Default Ctor instantiated");
  }

  TEST_FUNC explicit constexpr DefaultCtorBlowsUp(int x)
      : value(x)
  {}

  int value;
};

struct DerivedFromAllocArgT : cuda::std::allocator_arg_t
{ // allocator_arg_t has an explicit default constructor
  TEST_FUNC constexpr DerivedFromAllocArgT()
      : cuda::std::allocator_arg_t()
  {}
};

struct Explicit
{
  int value;
  TEST_FUNC constexpr explicit Explicit(int x)
      : value(x)
  {}
};

TEST_FUNC constexpr void test_uses_allocator_sfinae_evaluation();

TEST_FUNC constexpr bool test()
{
  A1<int> alloc{5};
  {
    cuda::std::tuple<constexpr_alloc_arg> from_int(cuda::std::allocator_arg, alloc, 7);
    assert(cuda::std::get<0>(from_int).value == 7);
  }
  {
    cuda::std::tuple<constexpr_alloc_last> last_from_int(cuda::std::allocator_arg, alloc, 9);
    assert(cuda::std::get<0>(last_from_int).value == 9);
  }
  {
    cuda::std::tuple<Explicit> t{cuda::std::allocator_arg, alloc, 42};
    assert(cuda::std::get<0>(t).value == 42);
  }
  {
    cuda::std::tuple<MoveOnly> t(cuda::std::allocator_arg, alloc, MoveOnly(0));
    assert(cuda::std::get<0>(t) == 0);
  }
  {
    using T = DefaultCtorBlowsUp<>;
    cuda::std::tuple<T> t(cuda::std::allocator_arg, alloc, T(42));
    assert(cuda::std::get<0>(t).value == 42);
  }
  {
    cuda::std::tuple<MoveOnly, MoveOnly> t(cuda::std::allocator_arg, alloc, MoveOnly(0), MoveOnly(1));
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t) == 1);
  }
  {
    using T = DefaultCtorBlowsUp<>;
    cuda::std::tuple<T, T> t(cuda::std::allocator_arg, alloc, T(42), T(43));
    assert(cuda::std::get<0>(t).value == 42);
    assert(cuda::std::get<1>(t).value == 43);
  }
  {
    cuda::std::tuple<MoveOnly, MoveOnly, MoveOnly> t(cuda::std::allocator_arg, alloc, MoveOnly(0), 1, 2);
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t) == 1);
    assert(cuda::std::get<2>(t) == 2);
  }
  {
    using T = DefaultCtorBlowsUp<>;
    cuda::std::tuple<T, T, T> t(cuda::std::allocator_arg, alloc, T(1), T(2), T(3));
    assert(cuda::std::get<0>(t).value == 1);
    assert(cuda::std::get<1>(t).value == 2);
    assert(cuda::std::get<2>(t).value == 3);
  }
  {
    cuda::std::tuple<int, constexpr_alloc_arg, constexpr_alloc_last> t(cuda::std::allocator_arg, alloc, 1, 2, 3);
    assert(cuda::std::get<0>(t) == 1);
    assert(cuda::std::get<1>(t).value == 2);
    assert(cuda::std::get<2>(t).value == 3);
  }
  {
    // A tag derived from allocator_arg_t still selects uses-allocator construction.
    DerivedFromAllocArgT tag{};
    cuda::std::tuple<int, constexpr_alloc_arg, constexpr_alloc_last> t(tag, alloc, 1, 2, 3);
    assert(cuda::std::get<0>(t) == 1);
    assert(cuda::std::get<1>(t).value == 2);
    assert(cuda::std::get<2>(t).value == 3);
  }
  // Stress test the SFINAE on the uses-allocator constructors and
  // ensure that the "reduced-arity-initialization" extension is not offered
  // for these constructors.
  test_uses_allocator_sfinae_evaluation();
  return true;
}

using Nothrow = cuda::std::tuple<nothrow_alloc_arg>;
static_assert(cuda::std::is_nothrow_constructible_v<Nothrow, cuda::std::allocator_arg_t, A1<int>, int>);

// Make sure the _Up... constructor SFINAEs out when the number of initializers
// is less that the number of elements in the tuple. Previously libc++ would
// offer these constructors as an extension but they broke conforming code.
TEST_FUNC constexpr void test_uses_allocator_sfinae_evaluation()
{
  using BadDefault = DefaultCtorBlowsUp<>;
  {
    using Tuple = cuda::std::tuple<MoveOnly, MoveOnly, BadDefault>;

    static_assert(!cuda::std::is_constructible<Tuple, cuda::std::allocator_arg_t, A1<int>, MoveOnly>::value);

    static_assert(
      cuda::std::is_constructible<Tuple, cuda::std::allocator_arg_t, A1<int>, MoveOnly, MoveOnly, BadDefault>::value);
  }
  {
    using Tuple = cuda::std::tuple<MoveOnly, MoveOnly, BadDefault, BadDefault>;

    static_assert(!cuda::std::is_constructible<Tuple, cuda::std::allocator_arg_t, A1<int>, MoveOnly, MoveOnly>::value);

    static_assert(
      cuda::std::
        is_constructible<Tuple, cuda::std::allocator_arg_t, A1<int>, MoveOnly, MoveOnly, BadDefault, BadDefault>::value);
  }
}

#if TEST_HAS_EXCEPTIONS() && _CCCL_HOST_COMPILATION()
void test_exceptions()
{
  using ThrowArg  = cuda::std::tuple<throw_on_alloc_arg>;
  using ThrowLast = cuda::std::tuple<throw_on_alloc_last>;
  static_assert(!cuda::std::is_nothrow_constructible_v<ThrowArg, cuda::std::allocator_arg_t, A1<int>, int>);
  static_assert(!cuda::std::is_nothrow_constructible_v<ThrowLast, cuda::std::allocator_arg_t, A1<int>, int>);

  try
  {
    [[maybe_unused]] ThrowArg t(cuda::std::allocator_arg, A1<int>{}, 1);
    assert(false);
  }
  catch (int)
  {}
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
  static_assert(test());
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()
  return 0;
}
