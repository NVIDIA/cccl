//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "reduce_determinism_fail_common.cuh"

int main()
{
  // expected-error {{"Only non-deterministic reductions are currently supported"}}
  reduce_with_determinism(::cuda::execution::determinism::gpu_to_gpu);

  return EXIT_FAILURE;
}
