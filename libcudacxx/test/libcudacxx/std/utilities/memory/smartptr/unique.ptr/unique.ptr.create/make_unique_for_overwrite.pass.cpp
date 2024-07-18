//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11

// template<class T>
//   constexpr unique_ptr<T> make_unique_for_overwrite(); // T is not array
//
// template<class T>
//   constexpr unique_ptr<T> make_unique_for_overwrite(size_t n); // T is U[]
//
// template<class T, class... Args>
//   unspecified make_unique_for_overwrite(Args&&...) = delete; // T is U[N]

#include <cuda/std/cassert>
#include <cuda/std/concepts>
// #include <cuda/std/cstring>
#include <cuda/std/__memory_>
#include <cuda/std/utility>

#include "test_macros.h"

template <class T, class... Args>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  HasMakeUniqueForOverwrite_,
  requires(Args&&... args)((cuda::std::make_unique_for_overwrite<T>(cuda::std::forward<Args>(args)...))));
template <class T, class... Args>
constexpr bool HasMakeUniqueForOverwrite = _LIBCUDACXX_FRAGMENT(HasMakeUniqueForOverwrite_, T, Args...);

struct Foo
{
  int i;
};

// template<class T>
//   constexpr unique_ptr<T> make_unique_for_overwrite();
static_assert(HasMakeUniqueForOverwrite<int>, "");
static_assert(HasMakeUniqueForOverwrite<Foo>, "");
static_assert(!HasMakeUniqueForOverwrite<int, int>, "");
static_assert(!HasMakeUniqueForOverwrite<Foo, Foo>, "");

// template<class T>
//   constexpr unique_ptr<T> make_unique_for_overwrite(size_t n);
static_assert(HasMakeUniqueForOverwrite<int[], cuda::std::size_t>, "");
static_assert(HasMakeUniqueForOverwrite<Foo[], cuda::std::size_t>, "");
static_assert(!HasMakeUniqueForOverwrite<int[]>, "");
static_assert(!HasMakeUniqueForOverwrite<Foo[]>, "");
static_assert(!HasMakeUniqueForOverwrite<int[], cuda::std::size_t, int>, "");
static_assert(!HasMakeUniqueForOverwrite<Foo[], cuda::std::size_t, int>, "");

// template<class T, class... Args>
//   unspecified make_unique_for_overwrite(Args&&...) = delete;
static_assert(!HasMakeUniqueForOverwrite<int[2]>, "");
static_assert(!HasMakeUniqueForOverwrite<int[2], cuda::std::size_t>, "");
static_assert(!HasMakeUniqueForOverwrite<int[2], int>, "");
static_assert(!HasMakeUniqueForOverwrite<int[2], int, int>, "");
static_assert(!HasMakeUniqueForOverwrite<Foo[2]>, "");
static_assert(!HasMakeUniqueForOverwrite<Foo[2], cuda::std::size_t>, "");
static_assert(!HasMakeUniqueForOverwrite<Foo[2], int>, "");
static_assert(!HasMakeUniqueForOverwrite<Foo[2], int, int>, "");

struct WithDefaultConstructor
{
  int i;
  __host__ __device__ constexpr WithDefaultConstructor()
      : i(5)
  {}
};

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  // single int
  {
    decltype(auto) ptr = cuda::std::make_unique_for_overwrite<int>();
    static_assert(cuda::std::same_as<cuda::std::unique_ptr<int>, decltype(ptr)>, "");
    // memory is available for write, otherwise constexpr test would fail
    *ptr = 5;
  }

  // unbounded array int[]
  {
    decltype(auto) ptrs = cuda::std::make_unique_for_overwrite<int[]>(3);
    static_assert(cuda::std::same_as<cuda::std::unique_ptr<int[]>, decltype(ptrs)>, "");

    // memory is available for write, otherwise constexpr test would fail
    ptrs[0] = 3;
    ptrs[1] = 4;
    ptrs[2] = 5;
  }

  // single with default constructor
  {
    decltype(auto) ptr = cuda::std::make_unique_for_overwrite<WithDefaultConstructor>();
    static_assert(cuda::std::same_as<cuda::std::unique_ptr<WithDefaultConstructor>, decltype(ptr)>, "");
    assert(ptr->i == 5);
  }

  // unbounded array with default constructor
  {
    decltype(auto) ptrs = cuda::std::make_unique_for_overwrite<WithDefaultConstructor[]>(3);
    static_assert(cuda::std::same_as<cuda::std::unique_ptr<WithDefaultConstructor[]>, decltype(ptrs)>, "");
    assert(ptrs[0].i == 5);
    assert(ptrs[1].i == 5);
    assert(ptrs[2].i == 5);
  }

  return true;
}

// The standard specifically says to use `new (p) T`, which means that we should pick up any
// custom in-class operator new if there is one.

STATIC_TEST_GLOBAL_VAR bool WithCustomNew_customNewCalled    = false;
STATIC_TEST_GLOBAL_VAR bool WithCustomNew_customNewArrCalled = false;

struct WithCustomNew
{
  __host__ __device__ static void* operator new(cuda::std::size_t n)
  {
    WithCustomNew_customNewCalled = true;
    return ::operator new(n);
    ;
  }

  __host__ __device__ static void* operator new[](cuda::std::size_t n)
  {
    WithCustomNew_customNewArrCalled = true;
    return ::operator new[](n);
  }
};

__host__ __device__ void testCustomNew()
{
  // single with custom operator new
  {
    decltype(auto) ptr = cuda::std::make_unique_for_overwrite<WithCustomNew>();
    static_assert(cuda::std::same_as<cuda::std::unique_ptr<WithCustomNew>, decltype(ptr)>, "");

    assert(WithCustomNew_customNewCalled);
    unused(ptr);
  }

  // unbounded array with custom operator new
  {
    decltype(auto) ptr = cuda::std::make_unique_for_overwrite<WithCustomNew[]>(3);
    static_assert(cuda::std::same_as<cuda::std::unique_ptr<WithCustomNew[]>, decltype(ptr)>, "");

    assert(WithCustomNew_customNewArrCalled);
    unused(ptr);
  }
}

int main(int, char**)
{
  test();
  testCustomNew();
#if TEST_STD_VER >= 2023
  static_assert(test());
#endif // TEST_STD_VER >= 2023

  return 0;
}
