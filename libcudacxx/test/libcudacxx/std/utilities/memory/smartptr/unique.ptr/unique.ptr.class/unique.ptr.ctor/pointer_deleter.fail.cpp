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

// unique_ptr<T, const D&>(pointer, D()) should not compile

// UNSUPPORTED: nvrtc

#include <cuda/std/__memory_>

struct Deleter
{
  void operator()(int* p) const
  {
    delete p;
  }
};

int main(int, char**)
{
  // expected-error@+1 {{call to deleted constructor of 'cuda::std::unique_ptr<int, const Deleter &>}}
  cuda::std::unique_ptr<int, const Deleter&> s((int*) nullptr, Deleter());

  return 0;
}
