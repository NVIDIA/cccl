//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <cuda/std/optional>

// template <class T>
// class optional
// {
// public:
//     typedef T value_type;
//     ...

#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::optional;

template <class Opt, class T>
__host__ __device__ void test()
{
  static_assert(cuda::std::is_same<typename Opt::value_type, T>::value, "");
}

int main(int, char**)
{
  test<optional<int>, int>();
  test<optional<const int>, const int>();
  test<optional<double>, double>();
  test<optional<const double>, const double>();

  return 0;
}
