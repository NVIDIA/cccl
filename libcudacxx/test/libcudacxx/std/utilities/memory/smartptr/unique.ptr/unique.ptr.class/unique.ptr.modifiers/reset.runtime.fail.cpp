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
// UNSUPPORTED: c++03

// unique_ptr

// test reset

// UNSUPPORTED: nvrtc

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "unique_ptr_test_helper.h"

int main(int, char**)
{
  {
    cuda::std::unique_ptr<A[]> p;
    p.reset(static_cast<B*>(nullptr)); // expected-error {{no matching member function for call to 'reset'}}
  }
  {
    cuda::std::unique_ptr<int[]> p;
    p.reset(static_cast<const int*>(nullptr)); // expected-error {{no matching member function for call to 'reset'}}
  }

  return 0;
}
