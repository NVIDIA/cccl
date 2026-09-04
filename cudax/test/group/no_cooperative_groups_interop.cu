//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#define _CUDAX_DISABLE_COOPERATIVE_GROUPS_INTEROP

#include <cuda/experimental/coop.cuh>
#include <cuda/experimental/group.cuh>

#include <c2h/catch2_test_helper.h>

C2H_TEST("cooperative-groups interop can be disabled", "[group]")
{
  STATIC_REQUIRE(!_CCCL_HAS_COOPERATIVE_GROUPS());
}
