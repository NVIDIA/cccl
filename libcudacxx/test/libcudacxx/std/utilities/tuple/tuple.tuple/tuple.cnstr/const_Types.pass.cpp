//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// explicit tuple(const T&...);

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

template <class...>
struct never
{
  enum
  {
    value = 0
  };
};

struct NoValueCtor
{
  STATIC_MEMBER_VAR(count, int)

  __host__ __device__ NoValueCtor()
      : id(++count())
  {}
  __host__ __device__ NoValueCtor(NoValueCtor const& other)
      : id(other.id)
  {
    ++count();
  }

  // The constexpr is required to make is_constructible instantiate this template.
  // The explicit is needed to test-around a similar bug with is_convertible.
  template <class T>
  __host__ __device__ constexpr explicit NoValueCtor(T)
  {
    static_assert(never<T>::value, "This should not be instantiated");
  }

  int id;
};

struct NoValueCtorEmpty
{
  __host__ __device__ NoValueCtorEmpty() {}
  __host__ __device__ NoValueCtorEmpty(NoValueCtorEmpty const&) {}

  template <class T>
  __host__ __device__ constexpr explicit NoValueCtorEmpty(T)
  {
    static_assert(never<T>::value, "This should not be instantiated");
  }
};

struct ImplicitCopy
{
  __host__ __device__ explicit ImplicitCopy(int) {}
  __host__ __device__ ImplicitCopy(ImplicitCopy const&) {}
};

// Test that tuple(cuda::std::allocator_arg, Alloc, Types const&...) allows implicit
// copy conversions in return value expressions.
__host__ __device__ cuda::std::tuple<ImplicitCopy> testImplicitCopy1()
{
  ImplicitCopy i(42);
  return {i};
}

__host__ __device__ cuda::std::tuple<ImplicitCopy> testImplicitCopy2()
{
  const ImplicitCopy i(42);
  return {i};
}

__host__ __device__ cuda::std::tuple<ImplicitCopy> testImplicitCopy3()
{
  const ImplicitCopy i(42);
  return i;
}

int main(int, char**)
{
  NoValueCtor::count() = 0;
  {
    // check that the literal '0' can implicitly initialize a stored pointer.
    cuda::std::tuple<int*> t = 0;
    assert(cuda::std::get<0>(t) == nullptr);
  }
  {
    cuda::std::tuple<int> t(2);
    assert(cuda::std::get<0>(t) == 2);
  }
  {
    constexpr cuda::std::tuple<int> t(2);
    static_assert(cuda::std::get<0>(t) == 2, "");
  }
  {
    constexpr cuda::std::tuple<int> t;
    static_assert(cuda::std::get<0>(t) == 0, "");
  }
  {
    cuda::std::tuple<int, char*> t(2, 0);
    assert(cuda::std::get<0>(t) == 2);
    assert(cuda::std::get<1>(t) == nullptr);
  }
  {
    constexpr cuda::std::tuple<int, char*> t(2, nullptr);
    static_assert(cuda::std::get<0>(t) == 2, "");
    static_assert(cuda::std::get<1>(t) == nullptr, "");
  }
  {
    cuda::std::tuple<int, char*> t(2, nullptr);
    assert(cuda::std::get<0>(t) == 2);
    assert(cuda::std::get<1>(t) == nullptr);
  }
  // cuda::std::string not supported
  /*
  {
      cuda::std::tuple<int, char*, cuda::std::string> t(2, nullptr, "text");
      assert(cuda::std::get<0>(t) == 2);
      assert(cuda::std::get<1>(t) == nullptr);
      assert(cuda::std::get<2>(t) == "text");
  }
  */
  // __tuple_leaf<T> uses is_constructible<T, U> to disable its explicit converting
  // constructor overload __tuple_leaf(U &&). Evaluating is_constructible can cause a compile error.
  // This overload is evaluated when __tuple_leafs copy or move ctor is called.
  // This checks that is_constructible is not evaluated when U == __tuple_leaf.
  {
    cuda::std::tuple<int, NoValueCtor, int, int> t(1, NoValueCtor(), 2, 3);
    assert(cuda::std::get<0>(t) == 1);
    assert(cuda::std::get<1>(t).id == 1);
    assert(cuda::std::get<2>(t) == 2);
    assert(cuda::std::get<3>(t) == 3);
  }
  {
    cuda::std::tuple<int, NoValueCtorEmpty, int, int> t(1, NoValueCtorEmpty(), 2, 3);
    assert(cuda::std::get<0>(t) == 1);
    assert(cuda::std::get<2>(t) == 2);
    assert(cuda::std::get<3>(t) == 3);
  }
  // extensions
  // cuda::std::string not supported
  /*
#ifdef _CUDA_STD_VERSION
  {
      cuda::std::tuple<int, char*, cuda::std::string> t(2);
      assert(cuda::std::get<0>(t) == 2);
      assert(cuda::std::get<1>(t) == nullptr);
      assert(cuda::std::get<2>(t) == "");
  }
  {
      cuda::std::tuple<int, char*, cuda::std::string> t(2, nullptr);
      assert(cuda::std::get<0>(t) == 2);
      assert(cuda::std::get<1>(t) == nullptr);
      assert(cuda::std::get<2>(t) == "");
  }
  {
      cuda::std::tuple<int, char*, cuda::std::string, double> t(2, nullptr, "text");
      assert(cuda::std::get<0>(t) == 2);
      assert(cuda::std::get<1>(t) == nullptr);
      assert(cuda::std::get<2>(t) == "text");
      assert(cuda::std::get<3>(t) == 0.0);
  }
#endif
  */

  return 0;
}
