//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// <memory>

// template <class Alloc>
// struct allocator_traits
// {
//     template <class T> using rebind_alloc  = Alloc::rebind<U>::other | Alloc<T, Args...>;
//     ...
// };

#include <cuda/std/__memory_>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
struct ReboundA
{};

template <class T>
struct A
{
  typedef T value_type;

  template <class U>
  struct rebind
  {
    typedef ReboundA<U> other;
  };
};

template <class T, class U>
struct ReboundB
{};

template <class T, class U>
struct B
{
  typedef T value_type;

  template <class V>
  struct rebind
  {
    typedef ReboundB<V, U> other;
  };
};

template <class T>
struct C
{
  typedef T value_type;
};

template <class T, class U>
struct D
{
  typedef T value_type;
};

template <class T>
struct E
{
  typedef T value_type;

  template <class U>
  struct rebind
  {
    typedef ReboundA<U> otter;
  };
};

#if !defined(TEST_COMPILER_MSVC_2017) // rebind is inaccessible
template <class T>
struct F
{
  typedef T value_type;

private:
  template <class>
  struct rebind
  {
    typedef void other;
  };
};

template <class T>
struct G
{
  typedef T value_type;
  template <class>
  struct rebind
  {
  private:
    typedef void other;
  };
};
#endif // !TEST_COMPILER_MSVC_2017

int main(int, char**)
{
  static_assert(
    (cuda::std::is_same<cuda::std::allocator_traits<A<char>>::rebind_alloc<double>, ReboundA<double>>::value), "");
  static_assert(
    (cuda::std::is_same<cuda::std::allocator_traits<B<int, char>>::rebind_alloc<double>, ReboundB<double, char>>::value),
    "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<C<char>>::rebind_alloc<double>, C<double>>::value), "");
  static_assert(
    (cuda::std::is_same<cuda::std::allocator_traits<D<int, char>>::rebind_alloc<double>, D<double, char>>::value), "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<E<char>>::rebind_alloc<double>, E<double>>::value), "");
#if !defined(TEST_COMPILER_MSVC_2017) // rebind is inaccessible
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<F<char>>::rebind_alloc<double>, F<double>>::value), "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<G<char>>::rebind_alloc<double>, G<double>>::value), "");
#endif // !TEST_COMPILER_MSVC_2017

  return 0;
}
