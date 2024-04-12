//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// ranges::advance
// Make sure we're SFINAE-friendly when the template argument constraints are not met.

#include <cuda/std/iterator>

class not_incrementable
{};

__host__ __device__ void proper_constraints()
{
  not_incrementable p{};
  cuda::std::ranges::advance(p, 5); // expected-error {{no matching function for call}}
  cuda::std::ranges::advance(p, p); // expected-error {{no matching function for call}}
  cuda::std::ranges::advance(p, 5, p); // expected-error {{no matching function for call}}
}

int main(int, char**)
{
  return 0;
}
