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

// Check that the nested types of cuda::std::allocator are provided:

// template <class T>
// class allocator
// {
// public:
//     typedef size_t    size_type;
//     typedef ptrdiff_t difference_type;
//     typedef T         value_type;
//
//     typedef T*        pointer;           // deprecated in C++17, removed in C++20
//     typedef T const*  const_pointer;     // deprecated in C++17, removed in C++20
//     typedef T&        reference;         // deprecated in C++17, removed in C++20
//     typedef T const&  const_reference;   // deprecated in C++17, removed in C++20
//     template< class U > struct rebind { typedef allocator<U> other; }; // deprecated in C++17, removed in C++20
//
//     typedef true_type propagate_on_container_move_assignment;
//     typedef true_type is_always_equal;
// ...
// };

// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/__memory_>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct U;

template <typename T>
__host__ __device__ void test()
{
  typedef cuda::std::allocator<T> Alloc;
  static_assert((cuda::std::is_same<typename Alloc::size_type, cuda::std::size_t>::value), "");
  static_assert((cuda::std::is_same<typename Alloc::difference_type, cuda::std::ptrdiff_t>::value), "");
  static_assert((cuda::std::is_same<typename Alloc::value_type, T>::value), "");
  static_assert(
    (cuda::std::is_same<typename Alloc::propagate_on_container_move_assignment, cuda::std::true_type>::value), "");
  static_assert((cuda::std::is_same<typename Alloc::is_always_equal, cuda::std::true_type>::value), "");

#if TEST_STD_VER <= 2017
  static_assert((cuda::std::is_same<typename Alloc::pointer, T*>::value), "");
  static_assert((cuda::std::is_same<typename Alloc::const_pointer, T const*>::value), "");
  static_assert((cuda::std::is_same<typename Alloc::reference, T&>::value), "");
  static_assert((cuda::std::is_same<typename Alloc::const_reference, T const&>::value), "");
  static_assert((cuda::std::is_same<typename Alloc::template rebind<U>::other, cuda::std::allocator<U>>::value), "");
#endif // TEST_STD_VER <= 2017
}

int main(int, char**)
{
  test<char>();
#ifdef _LIBCUDACXX_VERSION
  test<char const>(); // extension
#endif
  return 0;
}
