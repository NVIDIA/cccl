//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// flags<Flags...> allows at most one overaligned_flag<N>.

#include <cuda/std/__simd_>

#include "test_macros.h"

namespace simd = cuda::std::simd;

int main(int, char**)
{
  [[maybe_unused]] auto bad = simd::flag_overaligned<8> | simd::flag_overaligned<16>; // expected-error
  return 0;
}
