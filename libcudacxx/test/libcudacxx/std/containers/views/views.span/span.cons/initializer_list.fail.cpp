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
//
// This constructor must NOT participate in overload resolution when:
//   - element_type is not const (dangling risk), or
//   - the initializer_list element type does not exactly match value_type (narrowing risk)

#include <cuda/std/span>

int main(int, char**)
{
  // Non-const element type: span<int> from initializer_list must fail
  {
    cuda::std::span<int> s{1, 2, 3}; // expected-error
  }

  // Non-const element type with static extent
  {
    cuda::std::span<int, 3> s{1, 2, 3}; // expected-error
  }

  // Narrowing: int literals to span<const bool> must fail (same_as<int, bool> is false)
  {
    cuda::std::span<const bool> s{1, 0, 1}; // expected-error
  }

  // Narrowing: double to span<const int> must fail (same_as<double, int> is false)
  {
    cuda::std::span<const int> s{1.0, 2.0}; // expected-error
  }

  // Narrowing: int to bool with static extent
  {
    cuda::std::span<const bool, 3> s{1, 0, 1}; // expected-error
  }

  // Pointer element type: span<const int*> must fail because is_const_v<const int*> is false
  // (const int* is "pointer to const int", the pointer itself is not const)
  {
    int x = 0;
    cuda::std::span<const int*> s{&x}; // expected-error
  }

  return 0;
}
