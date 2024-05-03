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
//     static constexpr pointer allocate(allocator_type& a, size_type n);
//     ...
// };

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <cuda/std/__memory_>
#include <cuda/std/cstddef>

template <class T>
struct A
{
  typedef T value_type;
  value_type* allocate(cuda::std::size_t n);
  value_type* allocate(cuda::std::size_t n, const void* p);
};

void f()
{
  A<int> a;
  cuda::std::allocator_traits<A<int>>::allocate(a, 10); // expected-warning {{ignoring return value of function declared
                                                        // with 'nodiscard' attribute}}
  cuda::std::allocator_traits<A<int>>::allocate(a, 10, nullptr); // expected-warning {{ignoring return value of function
                                                                 // declared with 'nodiscard' attribute}}
}
