//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// overaligned_flag<N> requires N to be a power of two.

#include <cuda/std/__simd_>

#include "test_macros.h"

int main(int, char**)
{
  [[maybe_unused]] auto bad = cuda::std::simd::flag_overaligned<3>; // expected-error
  return 0;
}
