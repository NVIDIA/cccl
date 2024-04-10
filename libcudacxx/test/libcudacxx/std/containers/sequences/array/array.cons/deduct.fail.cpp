//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>
// UNSUPPORTED: c++03, c++11, c++14

// template <class T, class... U>
//   array(T, U...) -> array<T, 1 + sizeof...(U)>;
//
//  Requires: (is_same_v<T, U> && ...) is true. Otherwise the program is ill-formed.

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "test_macros.h"

int main(int, char**)
{
  {
    cuda::std::array arr{1, 2, 3L}; // expected-error {{no viable constructor or deduction guide for deduction of
                                    // template arguments of 'array'}}
  }

  return 0;
}
