//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: libcpp-has-no-threads

// Disable this test with nvcc 13.5 on windows, due to fabric.wait.sync_restrict::reads deprecation warnings promoted to
// errors.
// UNSUPPORTED: windows && nvcc-13.5

// <cuda/ptxs/fabric_wait.h>

#include <cuda/ptxs/fabric_wait.h>

#include "generated/fabric_wait.h"

int main(int, char**)
{
  return 0;
}
