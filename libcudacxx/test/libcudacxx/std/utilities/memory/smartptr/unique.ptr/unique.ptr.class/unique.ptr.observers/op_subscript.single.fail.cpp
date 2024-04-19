//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// <memory>

// unique_ptr

// test op[](size_t)

// UNSUPPORTED: nvrtc

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

int main(int, char**)
{
  cuda::std::unique_ptr<int> p(new int[3]);
  cuda::std::unique_ptr<int> const& cp = p;
  p[0]; // expected-error {{type 'cuda::std::unique_ptr<int>' does not provide a subscript operator}}
  cp[1]; // expected-error {{type 'const cuda::std::unique_ptr<int>' does not provide a subscript operator}}

  return 0;
}
