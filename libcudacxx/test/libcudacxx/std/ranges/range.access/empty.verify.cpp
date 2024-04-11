//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: nvrtc
// UNSUPPORTED: msvc-19.16

// std::ranges::empty

#include <cuda/std/ranges>

extern int arr[];

// Verify that for an array of unknown bound `ranges::empty` is ill-formed.
__host__ __device__ void test()
{
  cuda::std::ranges::empty(arr);
  // expected-error-re@-1 {{{{no matching function for call to object of type 'const (std::ranges::)?__empty::__fn'}}}}
}

int main(int, char**)
{
  return 0;
}
