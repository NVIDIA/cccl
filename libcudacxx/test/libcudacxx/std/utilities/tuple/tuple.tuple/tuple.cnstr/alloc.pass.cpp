//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc>
//   explicit(see-below) tuple(allocator_arg_t, const Alloc& a);

// NOTE: this constructor does not currently support tags derived from
// allocator_arg_t because libc++ has to deduce the parameter as a template
// argument. See PR27684 (https://bugs.llvm.org/show_bug.cgi?id=27684)

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "../alloc_constexpr_types.h"
#include "allocators.h"
#include "DefaultOnly.h"
#include "test_macros.h"

template <class T = void>
struct NonDefaultConstructible
{
  TEST_FUNC constexpr NonDefaultConstructible()
  {
    static_assert(!cuda::std::is_same<T, T>::value, "Default Ctor instantiated");
  }

  TEST_FUNC explicit constexpr NonDefaultConstructible(int) {}
};

TEST_FUNC constexpr bool test()
{
  A1<int> alloc{5};

  {
    [[maybe_unused]] cuda::std::tuple<> empty(cuda::std::allocator_arg, alloc);
  }
  {
    cuda::std::array<int, 0> empty_array{};
    [[maybe_unused]] cuda::std::tuple<> from_empty_array(cuda::std::allocator_arg, alloc, empty_array);
  }
  {
    cuda::std::tuple<constexpr_alloc_arg> def(cuda::std::allocator_arg, alloc);
    assert(cuda::std::get<0>(def).value == 1);
  }
  {
    cuda::std::tuple<constexpr_alloc_last> last_def(cuda::std::allocator_arg, alloc);
    assert(cuda::std::get<0>(last_def).value == 2);
  }
  {
    cuda::std::tuple<int> t(cuda::std::allocator_arg, alloc);
    assert(cuda::std::get<0>(t) == 0);
  }
  {
    cuda::std::tuple<int, constexpr_alloc_arg, constexpr_alloc_last> t(cuda::std::allocator_arg, alloc);
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t).value == 1);
    assert(cuda::std::get<2>(t).value == 2);
  }
  {
    // A2 is not convertible to the element allocator type, so the uses-allocator constructors are not selected.
    cuda::std::tuple<int, constexpr_alloc_arg, constexpr_alloc_last> t(cuda::std::allocator_arg, A2<int>{5});
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t).value == 0);
    assert(cuda::std::get<2>(t).value == 0);
  }
  {
    // The uses-allocator default constructor must not be instantiated when it is not selected.
    using T = NonDefaultConstructible<>;
    T v(42);
    [[maybe_unused]] cuda::std::tuple<T, T> ignored(v, v);
    [[maybe_unused]] cuda::std::tuple<T, T> from_ints(42, 42);
  }
  return true;
}

using Nothrow = cuda::std::tuple<nothrow_alloc_arg>;
static_assert(cuda::std::is_nothrow_constructible_v<Nothrow, cuda::std::allocator_arg_t, A1<int>>);

TEST_FUNC void test_runtime()
{
  // DefaultOnly has a user-provided destructor and updates a static counter.
  DefaultOnly::count() = 0;
  cuda::std::tuple<DefaultOnly> t(cuda::std::allocator_arg, A1<int>());
  assert(cuda::std::get<0>(t) == DefaultOnly());
}

#if TEST_HAS_EXCEPTIONS() && _CCCL_HOST_COMPILATION()
void test_exceptions()
{
  using ThrowArg        = cuda::std::tuple<throw_on_alloc_arg>;
  using ThrowLast       = cuda::std::tuple<throw_on_alloc_last>;
  using FromComplexLast = cuda::std::tuple<throw_on_alloc_last, throw_on_alloc_last>;
  static_assert(!cuda::std::is_nothrow_constructible_v<ThrowArg, cuda::std::allocator_arg_t, A1<int>>);
  static_assert(!cuda::std::is_nothrow_constructible_v<ThrowLast, cuda::std::allocator_arg_t, A1<int>>);

  try
  {
    [[maybe_unused]] ThrowArg t(cuda::std::allocator_arg, A1<int>{});
    assert(false);
  }
  catch (int)
  {}

  try
  {
    [[maybe_unused]] FromComplexLast t(cuda::std::allocator_arg, A1<int>{});
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
  test_runtime();
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()
  return 0;
}
