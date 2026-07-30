//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// gcc-10 segfaults with any use of constant_wrapper, gcc-11 fails to evaluate:
//   typename decltype(__cw_fixed_value(_Xp))::type
// UNSUPPORTED: gcc-10 || gcc-11

// todo(dabayer): Find a way to make this work for nvrtc.
// nvrtc doesn't allow accessing the static constexpr const auto& value member.
// UNSUPPORTED: nvrtc

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
