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
//     typedef Alloc::pointer | value_type* pointer;
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
  typedef T value_type;
  typedef Ptr<T> pointer;
};

template <class T>
struct B
{
  typedef T value_type;
};

#if !defined(TEST_COMPILER_MSVC_2017) // pointer is inaccessible
template <class T>
struct C
{
  typedef T value_type;

private:
  typedef void pointer;
};
#endif // !TEST_COMPILER_MSVC_2017

int main(int, char**)
{
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<A<char>>::pointer, Ptr<char>>::value), "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<B<char>>::pointer, char*>::value), "");
#if !defined(TEST_COMPILER_MSVC_2017) // pointer is inaccessible
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<C<char>>::pointer, char*>::value), "");
#endif // !TEST_COMPILER_MSVC_2017

  return 0;
}
