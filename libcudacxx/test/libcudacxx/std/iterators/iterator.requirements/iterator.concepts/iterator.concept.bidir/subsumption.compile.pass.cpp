//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: nvcc-12.0 || nvcc-12.1 || nvcc-12.2 || nvcc-12.3
// nvbug 3885350

// template<class T>
// concept bidirectional_iterator;

#include <cuda/std/concepts>
#include <cuda/std/iterator>

template <cuda::std::forward_iterator I>
__host__ __device__ constexpr bool check_subsumption()
{
  return false;
}

template <cuda::std::bidirectional_iterator>
__host__ __device__ constexpr bool check_subsumption()
{
  return true;
}

static_assert(check_subsumption<int*>());

int main(int, char**)
{
  return 0;
}
