//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70

// clang-cuda < 20 errors out with "fatal error: error in backend: Cannot cast between two non-generic address spaces"
// XFAIL: clang-14 && !nvcc
// XFAIL: clang-15 && !nvcc
// XFAIL: clang-16 && !nvcc
// XFAIL: clang-17 && !nvcc
// XFAIL: clang-18 && !nvcc
// XFAIL: clang-19 && !nvcc

#include "pipeline_memcpy_async_thread_scope_generic.h"

int main(int argc, char** argv)
{
  test_select_source<cuda::thread_scope_block, int32_t>();

  return 0;
}
