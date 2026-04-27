//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// basic_vec<T, Abi> requires T to be a vectorizable type.
// bool is explicitly excluded by the standard.

#include <cuda/std/__simd_>

#include "test_macros.h"

int main(int, char**)
{
  using Vec = cuda::std::simd::basic_vec<bool, cuda::std::simd::fixed_size<4>>;
  Vec vec(true); // expected-error
  return 0;
}
