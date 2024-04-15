//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

// <memory>
//
// template <class Alloc>
// struct allocator_traits
// {
//     typedef Alloc                        allocator_type;
//     typedef typename allocator_type::value_type
//                                          value_type;
//
//     typedef Alloc::pointer | value_type* pointer;
//     typedef Alloc::const_pointer
//           | pointer_traits<pointer>::rebind<const value_type>
//                                          const_pointer;
//     typedef Alloc::void_pointer
//           | pointer_traits<pointer>::rebind<void>
//                                          void_pointer;
//     typedef Alloc::const_void_pointer
//           | pointer_traits<pointer>::rebind<const void>
//                                          const_void_pointer;

template <typename Alloc>
__host__ __device__ void test_pointer()
{
  typename cuda::std::allocator_traits<Alloc>::pointer vp;
  typename cuda::std::allocator_traits<Alloc>::const_pointer cvp;

  unused(vp); // Prevent unused warning
  unused(cvp); // Prevent unused warning

  static_assert(cuda::std::is_same<bool, decltype(vp == vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp != vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp > vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp >= vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp < vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp <= vp)>::value, "");

  static_assert(cuda::std::is_same<bool, decltype(vp == cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp == vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp != cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp != vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp > cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp > vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp >= cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp >= vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp < cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp < vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp <= cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp <= vp)>::value, "");

  static_assert(cuda::std::is_same<bool, decltype(cvp == cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp != cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp > cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp >= cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp < cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp <= cvp)>::value, "");
}

template <typename Alloc>
__host__ __device__ void test_void_pointer()
{
  typename cuda::std::allocator_traits<Alloc>::void_pointer vp;
  typename cuda::std::allocator_traits<Alloc>::const_void_pointer cvp;

  unused(vp); // Prevent unused warning
  unused(cvp); // Prevent unused warning

  static_assert(cuda::std::is_same<bool, decltype(vp == vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp != vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp > vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp >= vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp < vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp <= vp)>::value, "");

  static_assert(cuda::std::is_same<bool, decltype(vp == cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp == vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp != cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp != vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp > cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp > vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp >= cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp >= vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp < cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp < vp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(vp <= cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp <= vp)>::value, "");

  static_assert(cuda::std::is_same<bool, decltype(cvp == cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp != cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp > cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp >= cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp < cvp)>::value, "");
  static_assert(cuda::std::is_same<bool, decltype(cvp <= cvp)>::value, "");
}

struct Foo
{
  int x;
};

int main(int, char**)
{
  test_pointer<cuda::std::allocator<char>>();
  test_pointer<cuda::std::allocator<int>>();
  test_pointer<cuda::std::allocator<Foo>>();

  test_void_pointer<cuda::std::allocator<char>>();
  test_void_pointer<cuda::std::allocator<int>>();
  test_void_pointer<cuda::std::allocator<Foo>>();

  return 0;
}
