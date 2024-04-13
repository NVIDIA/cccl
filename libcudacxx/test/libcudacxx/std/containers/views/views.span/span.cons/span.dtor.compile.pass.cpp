//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11

// <span>

// ~span() = default;

#include <cuda/std/span>
#include <cuda/std/type_traits>

template <class T>
__host__ __device__ constexpr bool testDestructor()
{
  static_assert(cuda::std::is_nothrow_destructible<T>::value, "");
  static_assert(cuda::std::is_trivially_destructible<T>::value, "");
  return true;
}

__host__ __device__ void test()
{
  testDestructor<cuda::std::span<int, 1>>();
  testDestructor<cuda::std::span<int>>();
}

int main(int, char**)
{
  return 0;
}
