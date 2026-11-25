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

// PTX 9.5 warns about fabric.wait.sync_restrict::reads and ptxas has no warning switch specific to it.
// ADDITIONAL_COMPILE_OPTIONS_PTXAS: --disable-warnings

// <cuda/ptxs/fabric_wait.h>

#include <cuda/ptxs/fabric_wait.h>

#include "generated/fabric_wait.h"

int main(int, char**)
{
  return 0;
}
