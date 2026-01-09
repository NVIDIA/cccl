//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// __is_fully_bounded_array_v

#include <cuda/std/__type_traits/is_fully_bounded_array.h>

static_assert(!cuda::std::__is_fully_bounded_array_v<int>);
static_assert(!cuda::std::__is_fully_bounded_array_v<int[]>);
static_assert(cuda::std::__is_fully_bounded_array_v<int[1]>);
static_assert(cuda::std::__is_fully_bounded_array_v<int[1][2]>);
static_assert(!cuda::std::__is_fully_bounded_array_v<int[][2]>);
static_assert(!cuda::std::__is_fully_bounded_array_v<int[][1][1]>);
static_assert(cuda::std::__is_fully_bounded_array_v<int[1][1][1]>);

int main(int, char**)
{
  return 0;
}
