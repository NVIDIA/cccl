// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <cuda/std/tuple>

// See llvm.org/PR20855

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

#if TEST_HAS_BUILTIN_IDENTIFIER(__reference_binds_to_temporary)
#  define ASSERT_REFERENCE_BINDS_TEMPORARY(...)     static_assert(__reference_binds_to_temporary(__VA_ARGS__), "")
#  define ASSERT_NOT_REFERENCE_BINDS_TEMPORARY(...) static_assert(!__reference_binds_to_temporary(__VA_ARGS__), "")
#else
#  define ASSERT_REFERENCE_BINDS_TEMPORARY(...)     static_assert(true, "")
#  define ASSERT_NOT_REFERENCE_BINDS_TEMPORARY(...) static_assert(true, "")
#endif

template <class Tp>
struct ConvertsTo
{
  using RawTp = typename cuda::std::remove_cv<typename cuda::std::remove_reference<Tp>::type>::type;

  __host__ __device__ operator Tp() const
  {
    return static_cast<Tp>(value);
  }

  mutable RawTp value;
};

struct Base
{};
struct Derived : Base
{};

static_assert(cuda::std::is_same<decltype("abc"), decltype(("abc"))>::value, "");
// cuda::std::string not supported
/*
ASSERT_REFERENCE_BINDS_TEMPORARY(cuda::std::string const&, decltype("abc"));
ASSERT_REFERENCE_BINDS_TEMPORARY(cuda::std::string const&, decltype(("abc")));
ASSERT_REFERENCE_BINDS_TEMPORARY(cuda::std::string const&, const char*&&);
*/
ASSERT_NOT_REFERENCE_BINDS_TEMPORARY(int&, const ConvertsTo<int&>&);
ASSERT_NOT_REFERENCE_BINDS_TEMPORARY(const int&, ConvertsTo<int&>&);
ASSERT_NOT_REFERENCE_BINDS_TEMPORARY(Base&, Derived&);

static_assert(cuda::std::is_constructible<int&, cuda::std::reference_wrapper<int>>::value, "");
static_assert(cuda::std::is_constructible<int const&, cuda::std::reference_wrapper<int>>::value, "");

template <class T>
struct CannotDeduce
{
  using type = T;
};

template <class... Args>
__host__ __device__ void F(typename CannotDeduce<cuda::std::tuple<Args...>>::type const&)
{}

__host__ __device__ void compile_tests()
{
  {
    F<int, int const&>(cuda::std::make_tuple(42, 42));
  }
  {
    F<int, int const&>(cuda::std::make_tuple<const int&, const int&>(42, 42));
    cuda::std::tuple<int, int const&> t(cuda::std::make_tuple<const int&, const int&>(42, 42));
  }
  // cuda::std::string not supported
  /*
  {
    auto fn = &F<int, cuda::std::string const&>;
    fn(cuda::std::tuple<int, cuda::std::string const&>(42, cuda::std::string("a")));
    fn(cuda::std::make_tuple(42, cuda::std::string("a")));
  }
  */
  {
    Derived d;
    cuda::std::tuple<Base&, Base const&> t(d, d);
  }
  {
    ConvertsTo<int&> ct;
    cuda::std::tuple<int, int&> t(42, ct);
  }
}

__host__ __device__ void allocator_tests()
{
  // cuda::std::allocator not supported
  // cuda::std::allocator<void> alloc;
  int x = 42;
  {
    cuda::std::tuple<int&> t(cuda::std::ref(x));
    assert(&cuda::std::get<0>(t) == &x);
    // cuda::std::allocator not supported
    /*
    cuda::std::tuple<int&> t1(cuda::std::allocator_arg, alloc, cuda::std::ref(x));
    assert(&cuda::std::get<0>(t1) == &x);
    */
  }
  {
    auto r         = cuda::std::ref(x);
    auto const& cr = r;
    cuda::std::tuple<int&> t(r);
    assert(&cuda::std::get<0>(t) == &x);
    cuda::std::tuple<int&> t1(cr);
    assert(&cuda::std::get<0>(t1) == &x);
    // cuda::std::allocator not supported
    /*
    cuda::std::tuple<int&> t2(cuda::std::allocator_arg, alloc, r);
    assert(&cuda::std::get<0>(t2) == &x);
    cuda::std::tuple<int&> t3(cuda::std::allocator_arg, alloc, cr);
    assert(&cuda::std::get<0>(t3) == &x);
    */
  }
  {
    cuda::std::tuple<int const&> t(cuda::std::ref(x));
    assert(&cuda::std::get<0>(t) == &x);
    cuda::std::tuple<int const&> t2(cuda::std::cref(x));
    assert(&cuda::std::get<0>(t2) == &x);
    // cuda::std::allocator not supported
    /*
    cuda::std::tuple<int const&> t3(cuda::std::allocator_arg, alloc, cuda::std::ref(x));
    assert(&cuda::std::get<0>(t3) == &x);
    cuda::std::tuple<int const&> t4(cuda::std::allocator_arg, alloc, cuda::std::cref(x));
    assert(&cuda::std::get<0>(t4) == &x);
    */
  }
  {
    auto r  = cuda::std::ref(x);
    auto cr = cuda::std::cref(x);
    cuda::std::tuple<int const&> t(r);
    assert(&cuda::std::get<0>(t) == &x);
    cuda::std::tuple<int const&> t2(cr);
    assert(&cuda::std::get<0>(t2) == &x);
    // cuda::std::allocator not supported
    /*
    cuda::std::tuple<int const&> t3(cuda::std::allocator_arg, alloc, r);
    assert(&cuda::std::get<0>(t3) == &x);
    cuda::std::tuple<int const&> t4(cuda::std::allocator_arg, alloc, cr);
    assert(&cuda::std::get<0>(t4) == &x);
    */
  }
}

int main(int, char**)
{
  compile_tests();
  allocator_tests();

  return 0;
}
