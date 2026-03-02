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
//     typedef Alloc::void_pointer
//           | pointer_traits<pointer>::rebind<void>
//                                          void_pointer;
//     ...
// };

#include <cuda/std/__memory_>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
struct Ptr
{};

template <class T>
struct A
{
  using value_type = T;
  using pointer    = Ptr<T>;
};

template <class T>
struct B
{
  using value_type = T;
};

template <class T>
struct CPtr
{};

template <class T>
struct C
{
  using value_type   = T;
  using void_pointer = CPtr<void>;
};

template <class T>
struct D
{
  using value_type = T;

private:
  using void_pointer = void;
};

int main(int, char**)
{
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<A<char>>::void_pointer, Ptr<void>>::value), "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<B<char>>::void_pointer, void*>::value), "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<C<char>>::void_pointer, CPtr<void>>::value), "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<D<char>>::void_pointer, void*>::value), "");

  return 0;
}
