//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// flags<Flags...> requires every type in the pack to be one of
// convert_flag, aligned_flag, or overaligned_flag<N>.

#include <cuda/std/__simd_>

#include "test_macros.h"

int main(int, char**)
{
  [[maybe_unused]] cuda::std::simd::flags<int> bad{}; // expected-error
  return 0;
}
