//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/utility>

// template <class T> typename add_rvalue_reference<T>::type declval() noexcept;

#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

class A
{
  __host__ __device__ A(const A&);
  __host__ __device__ A& operator=(const A&);
};

int main(int, char**)
{
  static_assert((cuda::std::is_same<decltype(cuda::std::declval<A>()), A&&>::value), "");

  return 0;
}
