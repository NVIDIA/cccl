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

// default_delete

// Test that default_delete's operator() requires a complete type

// UNSUPPORTED: nvrtc

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

struct A;

int main(int, char**)
{
  cuda::std::default_delete<A> d;
  A* p = 0;
  d(p);

  return 0;
}
