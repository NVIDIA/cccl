//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: msvc-19.16

// template<class T>
// concept borrowed_range;

#include <cuda/std/ranges>

template <cuda::std::ranges::range R>
__host__ __device__ consteval bool check_subsumption()
{
  return false;
}

template <cuda::std::ranges::borrowed_range R>
__host__ __device__ consteval bool check_subsumption()
{
  return true;
}

static_assert(check_subsumption<int (&)[8]>());

int main(int, char**)
{
  return 0;
}
