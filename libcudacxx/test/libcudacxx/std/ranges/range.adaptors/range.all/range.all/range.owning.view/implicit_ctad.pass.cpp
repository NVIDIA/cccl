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

// owning_view

// Make sure that the implicitly-generated CTAD works.

#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"

struct Range
{
  __host__ __device__ int* begin();
  __host__ __device__ int* end();
};

int main(int, char**)
{
  Range r;
  cuda::std::ranges::owning_view view{cuda::std::move(r)};
  unused(view);
  ASSERT_SAME_TYPE(decltype(view), cuda::std::ranges::owning_view<Range>);

  return 0;
}
