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
//     template <class T> using rebind_traits = allocator_traits<rebind_alloc<T>>;
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
  using value_type = T;

  template <class U>
  struct rebind
  {
    using other = ReboundA<U>;
  };
};

template <class T, class U>
struct ReboundB
{};

template <class T, class U>
struct B
{
  using value_type = T;

  template <class V>
  struct rebind
  {
    using other = ReboundB<V, U>;
  };
};

template <class T>
struct C
{
  using value_type = T;
};

template <class T, class U>
struct D
{
  using value_type = T;
};

template <class T>
struct E
{
  using value_type = T;

  template <class U>
  struct rebind
  {
    using otter = ReboundA<U>;
  };
};

int main(int, char**)
{
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<A<char>>::rebind_traits<double>,
                                    cuda::std::allocator_traits<ReboundA<double>>>::value),
                "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<B<int, char>>::rebind_traits<double>,
                                    cuda::std::allocator_traits<ReboundB<double, char>>>::value),
                "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<C<char>>::rebind_traits<double>,
                                    cuda::std::allocator_traits<C<double>>>::value),
                "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<D<int, char>>::rebind_traits<double>,
                                    cuda::std::allocator_traits<D<double, char>>>::value),
                "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<E<char>>::rebind_traits<double>,
                                    cuda::std::allocator_traits<E<double>>>::value),
                "");

  return 0;
}
