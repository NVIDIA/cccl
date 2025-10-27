//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/binutils.cuh>

#if _CCCL_HAS_INCLUDE(<nv_decode.h>)
#  error "This test requires <nv_decode.h> not to be findable in PATH."
#endif

int main(int, char**)
{
  return 0;
}
