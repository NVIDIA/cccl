//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: libcpp-no-exceptions
// UNSUPPORTED: nvrtc

// explicit bad_expected_access(E e);

// Effects: Initializes unex with cuda::std::move(e).

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

// test explicit
static_assert(cuda::std::convertible_to<int, int>, "");
static_assert(!cuda::std::convertible_to<int, cuda::std::bad_expected_access<int>>, "");

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (cuda::std::bad_expected_access<MoveOnly> b(MoveOnly{3}); assert(b.error().get() == 3);))

  return 0;
}
