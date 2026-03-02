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
//     typedef Alloc::difference_type
//           | pointer_traits<pointer>::difference_type         difference_type;
//     ...
// };

#include <cuda/std/__memory_>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
struct A
{
  using value_type      = T;
  using difference_type = short;
};

template <class T>
struct B
{
  using value_type = T;
};

template <class T>
struct C
{
  using value_type = T;
  struct pointer
  {};
  struct const_pointer
  {};
  struct void_pointer
  {};
  struct const_void_pointer
  {};
};

template <class T>
struct D
{
  using value_type = T;

private:
  using difference_type = void;
};

namespace cuda::std
{
template <>
struct pointer_traits<C<char>::pointer>
{
  using pointer         = C<char>::pointer;
  using element_type    = char;
  using difference_type = signed char;
};
} // namespace cuda::std

int main(int, char**)
{
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<A<char>>::difference_type, short>::value), "");
  static_assert(
    (cuda::std::is_same<cuda::std::allocator_traits<B<char>>::difference_type, cuda::std::ptrdiff_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<C<char>>::difference_type, signed char>::value), "");
  static_assert(
    (cuda::std::is_same<cuda::std::allocator_traits<D<char>>::difference_type, cuda::std::ptrdiff_t>::value), "");

  return 0;
}
