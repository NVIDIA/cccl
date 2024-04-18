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
//     typedef Alloc::propagate_on_container_swap
//           | false_type                   propagate_on_container_swap;
//     ...
// };

#include <cuda/std/__memory_>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
struct A
{
  typedef T value_type;
  typedef cuda::std::true_type propagate_on_container_swap;
};

template <class T>
struct B
{
  typedef T value_type;
};

#if !defined(TEST_COMPILER_MSVC_2017) // propagate_on_container_swap is inaccessible
template <class T>
struct C
{
  typedef T value_type;

private:
  typedef cuda::std::true_type propagate_on_container_swap;
};
#endif // !TEST_COMPILER_MSVC_2017

int main(int, char**)
{
  static_assert(
    (cuda::std::is_same<cuda::std::allocator_traits<A<char>>::propagate_on_container_swap, cuda::std::true_type>::value),
    "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<B<char>>::propagate_on_container_swap,
                                    cuda::std::false_type>::value),
                "");
#if !defined(TEST_COMPILER_MSVC_2017) // propagate_on_container_swap is inaccessible
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<C<char>>::propagate_on_container_swap,
                                    cuda::std::false_type>::value),
                "");
#endif // !TEST_COMPILER_MSVC_2017

  return 0;
}
