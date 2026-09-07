//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// basic_mask<Bytes, Abi> requires Bytes to map to a valid integer type (1, 2, 4, 8, or 16 with __int128).

#include <cuda/std/__simd_>

#include "test_macros.h"

int main(int, char**)
{
  using Mask = cuda::std::simd::basic_mask<3, cuda::std::simd::fixed_size<4>>;
  Mask mask(true); // expected-error
  return 0;
}
