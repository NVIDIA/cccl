//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

//  template <pair-like P> constexpr const pair& operator=(P&&) const;  // since C++23

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
// #include <cuda/std/string>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct ConstAssignable
{
  mutable int* ptr  = nullptr;
  ConstAssignable() = default;
  TEST_FUNC constexpr ConstAssignable(int* p)
      : ptr(p)
  {} // enable `subrange::operator pair-like`
  TEST_FUNC constexpr ConstAssignable const& operator=(ConstAssignable const& other) const
  {
    ptr = other.ptr;
    return *this;
  }

  constexpr ConstAssignable(ConstAssignable const&) = default; // defeat -Wdeprecated-copy
#if TEST_STD_VER >= 2020 || !TEST_COMPILER(MSVC)
  constexpr ConstAssignable& operator=(ConstAssignable const&) = default; // defeat -Wdeprecated-copy
#endif // TEST_STD_VER >=2020 || !TEST_COMPILER(MSVC)
};

struct ConstAssignableInt
{
  mutable int val      = 0;
  ConstAssignableInt() = default;
  TEST_FUNC constexpr ConstAssignableInt const& operator=(int v) const
  {
    val = v;
    return *this;
  }
};

struct NoCopy
{
  NoCopy()                         = default;
  NoCopy(NoCopy const&)            = delete;
  NoCopy(NoCopy&&)                 = default;
  NoCopy& operator=(NoCopy const&) = delete;
  TEST_FUNC constexpr NoCopy const& operator=(NoCopy&&) const
  {
    return *this;
  }
};

TEST_FUNC constexpr bool test()
{
  // Make sure assignment works from array and tuple
  {
    // Check from cuda::std::array
    {
      int x = 91, y = 92;
      cuda::std::array<int, 2> a          = {1, 2};
      cuda::std::pair<int&, int&> const p = {x, y};
      decltype(auto) result               = (p = a);
      static_assert(cuda::std::is_same_v<cuda::std::pair<int&, int&> const&, decltype(result)>);
      assert(&result == &p);
      assert(x == 1);
      assert(y == 2);
      // too small
      static_assert(!cuda::std::is_assignable_v<cuda::std::pair<int&, int&> const&, cuda::std::array<int, 1>>);
      static_assert(cuda::std::is_assignable_v<cuda::std::pair<int&, int&> const&, cuda::std::array<int, 2>>);
      // too large
      static_assert(!cuda::std::is_assignable_v<cuda::std::pair<int&, int&> const&, cuda::std::array<int, 3>>);
    }

    // Check from cuda::std::tuple
    {
      int x = 91, y = 92;
      cuda::std::tuple<int, int> a        = {1, 2};
      cuda::std::pair<int&, int&> const p = {x, y};
      decltype(auto) result               = (p = a);
      static_assert(cuda::std::is_same_v<cuda::std::pair<int&, int&> const&, decltype(result)>);
      assert(&result == &p);
      assert(x == 1);
      assert(y == 2);
      static_assert(!cuda::std::is_assignable_v<cuda::std::pair<int&, int&> const&, cuda::std::tuple<int>>); // too
                                                                                                             // small
      static_assert(
        cuda::std::is_assignable_v<cuda::std::pair<int&, int&> const&, cuda::std::tuple<int, int>>); // works (test the
                                                                                                     // test)
      static_assert(
        !cuda::std::is_assignable_v<cuda::std::pair<int&, int&> const&, cuda::std::tuple<int, int, int>>); // too large
    }

#ifdef COMPILER_BUG // FIXME(miscco): For whatever reason the conversion is not taken and all compiler fail
#  if _CCCL_HAS_CONCEPTS()
    // Make sure it works for ranges::subrange. This actually deserves an explanation: even though
    // the assignment operator explicitly excludes ranges::subrange specializations, such assignments
    // end up working because of ranges::subrange's implicit conversion to pair-like types.
    // This test ensures that the interoperability works as intended.
    {
      int data[] = {1, 2, 3, 4, 5};
      cuda::std::ranges::subrange<int*> a(data);
      cuda::std::pair<ConstAssignable, ConstAssignable> const p;
      decltype(auto) result = (p = a);
      static_assert(cuda::std::is_same_v<cuda::std::pair<ConstAssignable, ConstAssignable> const&, decltype(result)>);
      assert(&result == &p);
      assert(p.first.ptr == data);
      assert(p.second.ptr == data + 5);
    }
#  endif // _CCCL_HAS_CONCEPTS()
#endif // Fails
  }

  // Make sure we allow element conversion from a pair-like
  {
    cuda::std::tuple<int, int> a = {1, 2};
    cuda::std::pair<ConstAssignableInt, ConstAssignableInt> const p;
    decltype(auto) result = (p = a);
    static_assert(
      cuda::std::is_same_v<cuda::std::pair<ConstAssignableInt, ConstAssignableInt> const&, decltype(result)>);
    assert(&result == &p);
    assert(p.first.val == 1);
    assert(p.second.val == 2);
    // first not convertible
    static_assert(!cuda::std::is_assignable_v<cuda::std::pair<ConstAssignableInt, ConstAssignableInt> const&,
                                              cuda::std::tuple<void*, int>>);
    // second not convertible
    static_assert(!cuda::std::is_assignable_v<cuda::std::pair<ConstAssignableInt, ConstAssignableInt> const&,
                                              cuda::std::tuple<int, void*>>);
    static_assert(cuda::std::is_assignable_v<cuda::std::pair<ConstAssignableInt, ConstAssignableInt> const&,
                                             cuda::std::tuple<int, int>>);
  }

  // Make sure we forward the pair-like elements
  {
    cuda::std::tuple<NoCopy, NoCopy> a;
    cuda::std::pair<NoCopy, NoCopy> const p;
    p = cuda::std::move(a);
  }

  return true;
}

int main(int, char**)
{
  test();
// gcc cannot have mutable member in constant expression
#if !TEST_COMPILER(GCC)
  static_assert(test());
#endif // !TEST_COMPILER(GCC)

  return 0;
}
