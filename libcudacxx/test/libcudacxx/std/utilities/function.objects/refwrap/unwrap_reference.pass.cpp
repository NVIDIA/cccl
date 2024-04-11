//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <functional>
//
// template <class T>
// struct unwrap_reference;
//
// template <class T>
// using unwrap_reference_t = typename unwrap_reference<T>::type;

// UNSUPPORTED: c++03, c++11, c++14, c++17

// #include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

template <typename T, typename Expected>
__host__ __device__ void check_equal()
{
  static_assert(cuda::std::is_same_v<typename cuda::std::unwrap_reference<T>::type, Expected>);
  static_assert(cuda::std::is_same_v<typename cuda::std::unwrap_reference<T>::type, cuda::std::unwrap_reference_t<T>>);
}

template <typename T>
__host__ __device__ void check()
{
  check_equal<T, T>();
  check_equal<T&, T&>();
  check_equal<T const, T const>();
  check_equal<T const&, T const&>();

  check_equal<cuda::std::reference_wrapper<T>, T&>();
  check_equal<cuda::std::reference_wrapper<T const>, T const&>();
}

struct T
{};

int main(int, char**)
{
  check<T>();
  check<int>();
  check<float>();

  check<T*>();
  check<int*>();
  check<float*>();

  return 0;
}
