//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class S, class I>
// concept sentinel_for;

#include <cuda/std/array>
#include <cuda/std/concepts>
#include <cuda/std/iterator>

// clang-format off
template<cuda::std::input_or_output_iterator, cuda::std::semiregular>
__host__ __device__ constexpr bool check_sentinel_subsumption() {
  return false;
}

template<class I, cuda::std::sentinel_for<I> >
__host__ __device__ constexpr bool check_subsumption() {
  return true;
}
// clang-format on

static_assert(check_subsumption<cuda::std::array<int, 4>::iterator, cuda::std::array<int, 4>::iterator>());

int main(int, char**)
{
  return 0;
}
