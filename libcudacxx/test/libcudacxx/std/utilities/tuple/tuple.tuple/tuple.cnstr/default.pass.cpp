//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// explicit(see-below) constexpr tuple();

// UNSUPPORTED: c++98, c++03

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "DefaultOnly.h"
#include "test_macros.h"

struct NoDefault
{
  NoDefault() = delete;
  __host__ __device__ explicit NoDefault(int) {}
};

struct NoExceptDefault
{
  NoExceptDefault() noexcept = default;
};

struct ThrowingDefault
{
  __host__ __device__ ThrowingDefault() {}
};

struct IllFormedDefault
{
  __host__ __device__ IllFormedDefault(int x)
      : value(x)
  {}
  template <bool Pred = false>
  __host__ __device__ constexpr IllFormedDefault()
  {
    static_assert(Pred, "The default constructor should not be instantiated");
  }
  int value;
};

int main(int, char**)
{
  {
    cuda::std::tuple<> t;
    unused(t);
  }
  {
    cuda::std::tuple<int> t;
    assert(cuda::std::get<0>(t) == 0);
  }
  {
    cuda::std::tuple<int, char*> t;
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t) == nullptr);
  }
  // cuda::std::string not supported
  /*
  {
      cuda::std::tuple<int, char*, cuda::std::string> t;
      assert(cuda::std::get<0>(t) == 0);
      assert(cuda::std::get<1>(t) == nullptr);
      assert(cuda::std::get<2>(t) == "");
  }
  {
      cuda::std::tuple<int, char*, cuda::std::string, DefaultOnly> t;
      assert(cuda::std::get<0>(t) == 0);
      assert(cuda::std::get<1>(t) == nullptr);
      assert(cuda::std::get<2>(t) == "");
      assert(cuda::std::get<3>(t) == DefaultOnly());
  }
  */
  {
    // See bug #21157.
    static_assert(!cuda::std::is_default_constructible<cuda::std::tuple<NoDefault>>(), "");
    static_assert(!cuda::std::is_default_constructible<cuda::std::tuple<DefaultOnly, NoDefault>>(), "");
    static_assert(!cuda::std::is_default_constructible<cuda::std::tuple<NoDefault, DefaultOnly, NoDefault>>(), "");
  }
  {
    static_assert(noexcept(cuda::std::tuple<NoExceptDefault>()), "");
    static_assert(noexcept(cuda::std::tuple<NoExceptDefault, NoExceptDefault>()), "");

#ifndef TEST_COMPILER_BROKEN_SMF_NOEXCEPT
    static_assert(!noexcept(cuda::std::tuple<ThrowingDefault, NoExceptDefault>()), "");
    static_assert(!noexcept(cuda::std::tuple<NoExceptDefault, ThrowingDefault>()), "");
    static_assert(!noexcept(cuda::std::tuple<ThrowingDefault, ThrowingDefault>()), "");
#endif // !TEST_COMPILER_BROKEN_SMF_NOEXCEPT
  }
  {
    constexpr cuda::std::tuple<> t;
    unused(t);
  }
  {
    constexpr cuda::std::tuple<int> t;
    assert(cuda::std::get<0>(t) == 0);
  }
  {
    constexpr cuda::std::tuple<int, char*> t;
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t) == nullptr);
  }
  {
    // Check that the SFINAE on the default constructor is not evaluated when
    // it isn't needed. If the default constructor is evaluated then this test
    // should fail to compile.
    IllFormedDefault v(0);
    cuda::std::tuple<IllFormedDefault> t(v);
    unused(t);
  }
  {
    struct Base
    {};
    struct Derived : Base
    {
    protected:
      Derived() = default;
    };
    static_assert(!cuda::std::is_default_constructible<cuda::std::tuple<Derived, int>>::value, "");
  }

  return 0;
}
