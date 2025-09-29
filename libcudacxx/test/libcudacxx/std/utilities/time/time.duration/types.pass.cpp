//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// Test nested types

// typedef Rep rep;
// typedef Period period;

#include <cuda/std/chrono>
#include <cuda/std/ratio>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using D = cuda::std::chrono::duration<long, cuda::std::ratio<3, 2>>;
  static_assert(cuda::std::is_same_v<D::rep, long>);
  static_assert(cuda::std::is_same_v<D::period, cuda::std::ratio<3, 2>>);

  return 0;
}
