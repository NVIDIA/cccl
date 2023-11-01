//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: c++20 && nvcc, c++20 && nvrtc
// nvbug 3885350

// cuda::std::forward_iterator;

#include <cuda/std/iterator>

#include <cuda/std/concepts>

// clang-format off
template<cuda::std::input_iterator>
__host__ __device__ constexpr bool check_subsumption() {
  return false;
}

template<cuda::std::forward_iterator>
__host__ __device__ constexpr bool check_subsumption() {
  return true;
}
// clang-format on

static_assert(check_subsumption<int*>());

int main(int, char**)
{
  return 0;
}
