//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <span>

// template <class _InitListValueType>
//   requires same_as<_InitListValueType, value_type> && is_const_v<element_type>
// constexpr span(initializer_list<_InitListValueType> __il);

#include <cuda/std/span>

int main(int, char**)
{
  // Non-const element type with dynamic extent: fails because is_const_v<int> is false
  {
    cuda::std::span<int> s{1, 2, 3}; // expected-error {{no matching constructor for initialization of 'cuda::std::span<int>'}}
  }

  // Non-const element type with dynamic extent (double): fails because is_const_v<double> is false
  {
    cuda::std::span<double> s{1.0, 2.0}; // expected-error {{no matching constructor for initialization of 'cuda::std::span<double>'}}
  }

  // Non-const element type with static extent: fails because is_const_v<int> is false
  {
    cuda::std::span<int, 3> s{1, 2, 3}; // expected-error {{no matching constructor for initialization of 'cuda::std::span<int, 3>'}}
  }

  // Narrowing: int to bool: fails because same_as<int, bool> is false
  {
    cuda::std::span<const bool> s{1, 0, 1}; // expected-error {{no matching constructor for initialization of 'cuda::std::span<const bool>'}}
  }

  // Narrowing: int to short: fails because same_as<int, short> is false
  {
    cuda::std::span<const short> s{1, 2, 3}; // expected-error {{no matching constructor for initialization of 'cuda::std::span<const short>'}}
  }

  // Narrowing: double to int: fails because same_as<double, int> is false
  {
    cuda::std::span<const int> s{1.0, 2.0}; // expected-error {{no matching constructor for initialization of 'cuda::std::span<const int>'}}
  }

  // Narrowing: int to bool with static extent: fails because same_as<int, bool> is false
  {
    cuda::std::span<const bool, 3> s{1, 0, 1}; // expected-error {{no matching constructor for initialization of 'cuda::std::span<const bool, 3>'}}
  }

  return 0;
}
