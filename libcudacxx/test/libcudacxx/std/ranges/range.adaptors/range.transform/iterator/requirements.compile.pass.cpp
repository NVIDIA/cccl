//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// The requirements for transform_view::<iterator>'s members.

#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

static_assert(cuda::std::ranges::bidirectional_range<cuda::std::ranges::transform_view<BidirectionalView, PlusOne>>);
static_assert(!cuda::std::ranges::bidirectional_range<cuda::std::ranges::transform_view<ForwardView, PlusOne>>);

static_assert(cuda::std::ranges::random_access_range<cuda::std::ranges::transform_view<RandomAccessView, PlusOne>>);
static_assert(!cuda::std::ranges::random_access_range<cuda::std::ranges::transform_view<BidirectionalView, PlusOne>>);

int main(int, char**)
{
  return 0;
}
