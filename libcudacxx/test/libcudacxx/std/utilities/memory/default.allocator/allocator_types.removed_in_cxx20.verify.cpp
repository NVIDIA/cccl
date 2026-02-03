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
//     using pointer         = T*;
//     using const_pointer   = const T*;
//     using reference       = typename add_lvalue_reference<T>::type;
//     using const_reference = typename add_lvalue_reference<const T>::type;
//
//     template <class U> struct rebind {using other = allocator<U>;};
// ...
// };

// UNSUPPORTED: c++17

#include <cuda/std/__memory_>

template <typename T>
__host__ __device__ void check()
{
  using AP  = typename cuda::std::allocator<T>::pointer; // expected-error 3 {{no type named 'pointer'}}
  using ACP = typename cuda::std::allocator<T>::const_pointer; // expected-error 3 {{no type named 'const_pointer'}}
  using AR  = typename cuda::std::allocator<T>::reference; // expected-error 3 {{no type named 'reference'}}
  using ACR = typename cuda::std::allocator<T>::const_reference; // expected-error 3 {{no type named 'const_reference'}}
  using ARO = typename cuda::std::allocator<T>::template rebind<int>::other; // expected-error 3 {{no member named
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
