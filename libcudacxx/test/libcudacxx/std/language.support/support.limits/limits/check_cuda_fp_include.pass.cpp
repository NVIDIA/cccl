//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Test that <cuda/std/limits> don't include any of the CUDA fp headers.

#include <cuda/std/limits>

#if defined(__CUDA_FP16_H__) || defined(__CUDA_BF16_H__) || defined(__CUDA_FP8_H__) || defined(__CUDA_FP6_H__) \
  || defined(__CUDA_FP4_H__)
#  error "any of the nvfp headers was included by <cuda/std/limits>"
#endif

int main(int, char**)
{
  return 0;
}
