//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <utility>

// template<class T, T N>
//   using make_integer_sequence = integer_sequence<T, 0, 1, ..., N-1>;

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda::std::is_same<cuda::std::make_integer_sequence<int, 0>, cuda::std::integer_sequence<int>>::value,
                "");
  static_assert(
    cuda::std::is_same<cuda::std::make_integer_sequence<int, 1>, cuda::std::integer_sequence<int, 0>>::value, "");
  static_assert(
    cuda::std::is_same<cuda::std::make_integer_sequence<int, 2>, cuda::std::integer_sequence<int, 0, 1>>::value, "");
  static_assert(
    cuda::std::is_same<cuda::std::make_integer_sequence<int, 3>, cuda::std::integer_sequence<int, 0, 1, 2>>::value, "");

  static_assert(cuda::std::is_same<cuda::std::make_integer_sequence<unsigned long long, 0>,
                                   cuda::std::integer_sequence<unsigned long long>>::value,
                "");
  static_assert(cuda::std::is_same<cuda::std::make_integer_sequence<unsigned long long, 1>,
                                   cuda::std::integer_sequence<unsigned long long, 0>>::value,
                "");
  static_assert(cuda::std::is_same<cuda::std::make_integer_sequence<unsigned long long, 2>,
                                   cuda::std::integer_sequence<unsigned long long, 0, 1>>::value,
                "");
  static_assert(cuda::std::is_same<cuda::std::make_integer_sequence<unsigned long long, 3>,
                                   cuda::std::integer_sequence<unsigned long long, 0, 1, 2>>::value,
                "");

  return 0;
}
