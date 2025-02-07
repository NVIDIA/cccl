//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cuda/std/mdspan>

#include <test_macros.h>

struct alignas(64) dummy
{
  int array[4] = {1, 2, 3, 4};
};

int main(int, char**)
{
// the alignment check is disabled when it is not possible to evaluate the alignment at compile time
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  using E = cuda::std::extents<size_t, 2>;
  using L = cuda::std::layout_right;
  using A = cuda::std::aligned_accessor<int, 64>;
  assert(alignof(dummy) == 64);
  dummy d;
  cuda::std::mdspan<int, E, L, A> md(static_cast<int*>(d.array) + 1, 2);
  unused(md(0));
  return 0;
#else
  return 1;
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED
}
