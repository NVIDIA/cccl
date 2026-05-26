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

// nvcc 12.0 segfaults.
// UNSUPPORTED: nvcc-12.0

// todo(dabayer): Find a way to make this work for nvrtc.
// nvrtc doesn't allow accessing the static constexpr const auto& value member.
// UNSUPPORTED: nvrtc

// REQUIRES: !c++17

// constant_wrapper

// The class template constant_wrapper aids in metaprogramming by ensuring that the
// evaluation of expressions comprised entirely of constant_wrapper are core constant
// expressions ([expr.const]), regardless of the context in which they appear. In particular,
// this enables use of constant_wrapper values that are passed as arguments to constexpr
// functions to be used in constant expressions.

#include <cuda/std/utility>

#include "test_macros.h"

TEST_FUNC constexpr auto initial_phase(auto quantity_1, auto quantity_2)
{
  return quantity_1 + quantity_2;
}

TEST_FUNC constexpr auto middle_phase(auto tbd)
{
  return tbd;
}

TEST_FUNC constexpr void profit() {}

TEST_FUNC void final_phase(auto gathered, auto available)
{
  if constexpr (gathered == available)
  {
    profit();
  }
}

TEST_FUNC void impeccable_underground_planning()
{
  auto gathered_quantity = middle_phase(initial_phase(cuda::std::__cw<42>, cuda::std::__cw<13>));
  static_assert(gathered_quantity == 55);
  auto all_available = cuda::std::__cw<55>;
  final_phase(gathered_quantity, all_available);
}

int main(int, char**)
{
  impeccable_underground_planning();
  return 0;
}
