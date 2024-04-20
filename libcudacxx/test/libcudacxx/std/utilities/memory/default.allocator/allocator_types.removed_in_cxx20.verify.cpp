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

// Check that the following nested types are removed in C++20:

// template <class T>
// class allocator
// {
// public:
//     typedef T*                                           pointer;
//     typedef const T*                                     const_pointer;
//     typedef typename add_lvalue_reference<T>::type       reference;
//     typedef typename add_lvalue_reference<const T>::type const_reference;
//
//     template <class U> struct rebind {typedef allocator<U> other;};
// ...
// };

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <cuda/std/__memory_>

template <typename T>
__host__ __device__ void check()
{
  typedef typename cuda::std::allocator<T>::pointer AP; // expected-error 3 {{no type named 'pointer'}}
  typedef typename cuda::std::allocator<T>::const_pointer ACP; // expected-error 3 {{no type named 'const_pointer'}}
  typedef typename cuda::std::allocator<T>::reference AR; // expected-error 3 {{no type named 'reference'}}
  typedef typename cuda::std::allocator<T>::const_reference ACR; // expected-error 3 {{no type named 'const_reference'}}
  typedef typename cuda::std::allocator<T>::template rebind<int>::other ARO; // expected-error 3 {{no member named
                                                                             // 'rebind'}}
}

__host__ __device__ void f()
{
  check<char>();
  check<char const>();
  check<void>();
}

int main(int, char**)
{
  return 0;
}
