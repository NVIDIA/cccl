//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: libcpp-has-no-threads

// <cuda/ptx>

#include <cuda/ptx>
#include <cuda/std/utility>

#include "nvrtc_workaround.h"
// above header needs to be included before the generated test header
#include "generated/mbarrier_arrive.h"
#include "generated/mbarrier_arrive_expect_tx.h"
#include "generated/mbarrier_arrive_no_complete.h"

int main(int, char**)
{
  return 0;
}
