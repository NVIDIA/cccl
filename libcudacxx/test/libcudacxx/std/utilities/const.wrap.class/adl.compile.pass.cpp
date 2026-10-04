//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo(dabayer): nvrtc doesn't support non-trivial types as static data members without -default-device, fails with:
//   A class static data member with non-const type is considered a host variable, and host variables are not allowed in
//   JIT mode. Consider using -default-device flag to process such data members as __device__ variables in JIT mode
// UNSUPPORTED: nvrtc

// NTTP may not have a class type in C++17.
// REQUIRES: !c++17

// constant_wrapper

// [Note 1: The unnamed second template parameter to constant_wrapper is present
// to aid argument-dependent lookup ([basic.lookup.argdep]) in finding overloads
// for which constant_wrapper's wrapped value is a suitable argument, but for which
// the constant_wrapper itself is not. - end note]

#include <cuda/std/utility>

#include "test_macros.h"

namespace MyNamespace
{
struct MyType
{};

TEST_FUNC void adl_function(MyType) {}
} // namespace MyNamespace

TEST_FUNC void test()
{
  cuda::std::__constant_wrapper<MyNamespace::MyType{}> cw_mt;
  adl_function(cw_mt);
}

int main(int, char**)
{
  return 0;
}
