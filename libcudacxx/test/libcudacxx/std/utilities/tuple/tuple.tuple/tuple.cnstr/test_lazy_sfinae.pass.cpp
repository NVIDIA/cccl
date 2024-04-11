//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: msvc, gcc-4.8

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

template <class ConstructFrom>
struct ConstructibleFromT
{
  ConstructibleFromT() = default;
  __host__ __device__ ConstructibleFromT(ConstructFrom v)
      : value(v)
  {}
  ConstructFrom value;
};

template <class AssertOn>
struct CtorAssertsT
{
  bool defaulted;
  __host__ __device__ CtorAssertsT()
      : defaulted(true)
  {}
  template <class T>
  __host__ __device__ constexpr CtorAssertsT(T)
      : defaulted(false)
  {
    static_assert(!cuda::std::is_same<T, AssertOn>::value, "");
  }
};

template <class AllowT, class AssertT>
struct AllowAssertT
{
  AllowAssertT() = default;
  __host__ __device__ AllowAssertT(AllowT) {}
  template <class U>
  __host__ __device__ constexpr AllowAssertT(U)
  {
    static_assert(!cuda::std::is_same<U, AssertT>::value, "");
  }
};

// Construct a tuple<T1, T2> from pair<int, int> where T1 and T2
// are not constructible from ints but T1 is constructible from cuda::std::pair.
// This considers the following constructors:
// (1) tuple(TupleLike) -> checks is_constructible<Tn, int>
// (2) tuple(UTypes...) -> checks is_constructible<T1, pair<int, int>>
//                            and is_default_constructible<T2>
// The point of this test is to ensure that the consideration of (1)
// short circuits before evaluating is_constructible<T2, int>, which
// will cause a static assertion.
__host__ __device__ void test_tuple_like_lazy_sfinae()
{
#if defined(_LIBCUDACXX_VERSION)
  // This test requires libc++'s reduced arity initialization.
  using T1 = ConstructibleFromT<cuda::std::pair<int, int>>;
  using T2 = CtorAssertsT<int>;
  cuda::std::pair<int, int> p(42, 100);
  cuda::std::tuple<T1, T2> t(p);
  assert(cuda::std::get<0>(t).value == p);
  assert(cuda::std::get<1>(t).defaulted);
#endif
}

struct NonConstCopyable
{
  NonConstCopyable() = default;
  __host__ __device__ explicit NonConstCopyable(int v)
      : value(v)
  {}
  NonConstCopyable(NonConstCopyable&)       = default;
  NonConstCopyable(NonConstCopyable const&) = delete;
  int value;
};

template <class T>
struct BlowsUpOnConstCopy
{
  BlowsUpOnConstCopy() = default;
  __host__ __device__ constexpr BlowsUpOnConstCopy(BlowsUpOnConstCopy const&)
  {
    static_assert(!cuda::std::is_same<T, T>::value, "");
  }
  BlowsUpOnConstCopy(BlowsUpOnConstCopy&) = default;
};

// Test the following constructors:
// (1) tuple(Types const&...)
// (2) tuple(UTypes&&...)
// Test that (1) short circuits before evaluating the copy constructor of the
// second argument. Constructor (2) should be selected.
__host__ __device__ void test_const_Types_lazy_sfinae()
{
  NonConstCopyable v(42);
  BlowsUpOnConstCopy<int> b;
  cuda::std::tuple<NonConstCopyable, BlowsUpOnConstCopy<int>> t(v, b);
  assert(cuda::std::get<0>(t).value == 42);
}

int main(int, char**)
{
  test_tuple_like_lazy_sfinae();
  test_const_Types_lazy_sfinae();

  return 0;
}
