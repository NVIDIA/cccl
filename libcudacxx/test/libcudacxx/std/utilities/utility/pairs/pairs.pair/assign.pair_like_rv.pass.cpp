//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

//  template <pair-like P> constexpr pair& operator=(P&&);  // since C++23

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
// #include <cuda/std/string>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct Assignable
{
  int* ptr     = nullptr;
  Assignable() = default;
  TEST_FUNC constexpr Assignable(int* p)
      : ptr(p)
  {} // enable `subrange::operator pair-like`
  constexpr Assignable& operator=(Assignable const&) = default;
};

struct NoCopy
{
  NoCopy()                         = default;
  NoCopy(NoCopy const&)            = delete;
  NoCopy(NoCopy&&)                 = default;
  NoCopy& operator=(NoCopy const&) = delete;
  NoCopy& operator=(NoCopy&&)      = default;
};

TEST_FUNC constexpr bool test()
{
  // Make sure assignment works from array and tuple
  {
    // Check from cuda::std::array
    {
      cuda::std::array<int, 2> a = {1, 2};
      cuda::std::pair<int, int> p;
      decltype(auto) result = (p = a);
      static_assert(cuda::std::is_same_v<cuda::std::pair<int, int>&, decltype(result)>);
      assert(&result == &p);
      assert(p.first == 1);
      assert(p.second == 2);
      static_assert(!cuda::std::is_assignable_v<cuda::std::pair<int, int>&, cuda::std::array<int, 1>>); // too small
      static_assert(cuda::std::is_assignable_v<cuda::std::pair<int, int>&, cuda::std::array<int, 2>>); // works (test
                                                                                                       // the test)
      static_assert(!cuda::std::is_assignable_v<cuda::std::pair<int, int>&, cuda::std::array<int, 3>>); // too large
    }

    // Check from cuda::std::tuple
    {
      cuda::std::tuple<int, int> a = {1, 2};
      cuda::std::pair<int, int> p;
      decltype(auto) result = (p = a);
      static_assert(cuda::std::is_same_v<cuda::std::pair<int, int>&, decltype(result)>);
      assert(&result == &p);
      assert(p.first == 1);
      assert(p.second == 2);
      // too small
      static_assert(!cuda::std::is_assignable_v<cuda::std::pair<int, int>&, cuda::std::tuple<int>>);
      static_assert(cuda::std::is_assignable_v<cuda::std::pair<int, int>&, cuda::std::tuple<int, int>>);
      // too large
      static_assert(!cuda::std::is_assignable_v<cuda::std::pair<int, int>&, cuda::std::tuple<int, int, int>>);
    }

#if _CCCL_HAS_CONCEPTS()
    // Make sure it works for ranges::subrange. This actually deserves an explanation: even though
    // the assignment operator explicitly excludes ranges::subrange specializations, such assignments
    // end up working because of ranges::subrange's implicit conversion to pair-like types.
    // This test ensures that the interoperability works as intended.
    {
      int data[] = {1, 2, 3, 4, 5};
      cuda::std::ranges::subrange<int*> a(data);
      cuda::std::pair<Assignable, Assignable> p;
      decltype(auto) result = (p = a);
      static_assert(cuda::std::is_same_v<cuda::std::pair<Assignable, Assignable>&, decltype(result)>);
      assert(&result == &p);
      assert(p.first.ptr == data);
      assert(p.second.ptr == data + 5);
    }
#endif // _CCCL_HAS_CONCEPTS()
  }

  // Make sure we allow element conversion from a pair-like
  {
    cuda::std::tuple<int, float> a = {34, 42.0f};
    cuda::std::pair<long, double> p;
    decltype(auto) result = (p = a);
    static_assert(cuda::std::is_same_v<cuda::std::pair<long, double>&, decltype(result)>);
    assert(&result == &p);
    assert(p.first == 34);
    assert(p.second == 42.0);
    // first not convertible
    static_assert(!cuda::std::is_assignable_v<cuda::std::pair<long, double>&, cuda::std::tuple<char*, double>>);
    // second not convertible
    static_assert(!cuda::std::is_assignable_v<cuda::std::pair<long, double>&, cuda::std::tuple<long, void*>>);
    static_assert(cuda::std::is_assignable_v<cuda::std::pair<long, double>&, cuda::std::tuple<long, double>>);
  }

  // Make sure we forward the pair-like elements
  {
    cuda::std::tuple<NoCopy, NoCopy> a;
    [[maybe_unused]] cuda::std::pair<NoCopy, NoCopy> p;
    p = cuda::std::move(a);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
