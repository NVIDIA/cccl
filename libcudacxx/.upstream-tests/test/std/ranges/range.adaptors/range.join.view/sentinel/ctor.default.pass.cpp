//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// sentinel() = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"
#include "../types.h"

__host__ __device__ constexpr bool test() {
  cuda::std::ranges::sentinel_t<cuda::std::ranges::join_view<CopyableParent>> sent;
  unused(sent);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
