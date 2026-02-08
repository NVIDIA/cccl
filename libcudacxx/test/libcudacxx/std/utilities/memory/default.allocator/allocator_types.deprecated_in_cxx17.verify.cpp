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

// Check that the following nested types are deprecated in C++17:

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

// REQUIRES: c++17

#include <cuda/std/__memory_>

__host__ __device__ void f()
{
  {
    using Pointer      = cuda::std::allocator<char>::pointer; // expected-warning {{'pointer' is deprecated}}
    using ConstPointer = cuda::std::allocator<char>::const_pointer; // expected-warning {{'const_pointer' is
                                                                    // deprecated}}
    using Reference      = cuda::std::allocator<char>::reference; // expected-warning {{'reference' is deprecated}}
    using ConstReference = cuda::std::allocator<char>::const_reference; // expected-warning {{'const_reference' is
                                                                        // deprecated}}
    using Rebind = cuda::std::allocator<char>::rebind<int>::other; // expected-warning {{'rebind<int>' is deprecated}}
  }
  {
    using Pointer      = cuda::std::allocator<char const>::pointer; // expected-warning {{'pointer' is deprecated}}
    using ConstPointer = cuda::std::allocator<char const>::const_pointer; // expected-warning {{'const_pointer' is
                                                                          // deprecated}}
    using Reference = cuda::std::allocator<char const>::reference; // expected-warning {{'reference' is deprecated}}
    using ConstReference = cuda::std::allocator<char const>::const_reference; // expected-warning {{'const_reference' is
                                                                              // deprecated}}
    using Rebind = cuda::std::allocator<char const>::rebind<int>::other; // expected-warning {{'rebind<int>' is
                                                                         // deprecated}}
  }
  {
    using Pointer      = cuda::std::allocator<void>::pointer; // expected-warning {{'pointer' is deprecated}}
    using ConstPointer = cuda::std::allocator<void>::const_pointer; // expected-warning {{'const_pointer' is
                                                                    // deprecated}}
    // reference and const_reference are not provided by cuda::std::allocator<void>
    using Rebind = cuda::std::allocator<void>::rebind<int>::other; // expected-warning {{'rebind<int>' is deprecated}}
  }
}

int main(int, char**)
{
  return 0;
}
