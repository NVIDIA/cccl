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
  typedef T value_type;
  typedef short difference_type;
};

template <class T>
struct B
{
  typedef T value_type;
};

template <class T>
struct C
{
  typedef T value_type;
  struct pointer
  {};
  struct const_pointer
  {};
  struct void_pointer
  {};
  struct const_void_pointer
  {};
};

#if !defined(TEST_COMPILER_MSVC_2017) // difference_type is inaccessible
template <class T>
struct D
{
  typedef T value_type;

private:
  typedef void difference_type;
};
#endif // !TEST_COMPILER_MSVC_2017

namespace cuda
{
namespace std
{

template <>
struct pointer_traits<C<char>::pointer>
{
  typedef C<char>::pointer pointer;
  typedef char element_type;
  typedef signed char difference_type;
};

} // namespace std
} // namespace cuda

int main(int, char**)
{
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<A<char>>::difference_type, short>::value), "");
  static_assert(
    (cuda::std::is_same<cuda::std::allocator_traits<B<char>>::difference_type, cuda::std::ptrdiff_t>::value), "");
  static_assert((cuda::std::is_same<cuda::std::allocator_traits<C<char>>::difference_type, signed char>::value), "");
#if !defined(TEST_COMPILER_MSVC_2017) // difference_type is inaccessible
  static_assert(
    (cuda::std::is_same<cuda::std::allocator_traits<D<char>>::difference_type, cuda::std::ptrdiff_t>::value), "");
#endif // !TEST_COMPILER_MSVC_2017

  return 0;
}
