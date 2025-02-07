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

int main(int, char**)
{
// the alignment check is disabled when it is not possible to evaluate the alignment at compile time
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  auto ptr = reinterpret_cast<int*>(0x1);
  cuda::std::aligned_accessor<int, 64> aligned;
  unused(aligned.offset(ptr, 0));
  return 0;
#else
  return 1;
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED
}
