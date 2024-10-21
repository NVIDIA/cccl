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

// clang-cuda errors out with "fatal error: error in backend: Cannot cast between two non-generic address spaces"
// XFAIL: clang && !nvcc

#include "pipeline_memcpy_async_thread_scope_thread.h"

int main(int argc, char** argv)
{
  test_select_source<uint32_t>();

  return 0;
}
