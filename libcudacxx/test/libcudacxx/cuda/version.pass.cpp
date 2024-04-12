//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/version>

static_assert(CCCL_MAJOR_VERSION == (CCCL_VERSION / 1000000), "");
static_assert(CCCL_MINOR_VERSION == (CCCL_VERSION / 1000 % 1000), "");
static_assert(CCCL_PATCH_VERSION == (CCCL_VERSION % 1000), "");

int main(int argc, char** argv)
{
  return 0;
}
