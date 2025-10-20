
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <random>

#include <cuda/__random/pcg_engine.h>

#include "../../std/random/engine/test_engine.h"

int main(int, char**)
{
  test_engine<cuda::pcg64_engine, 11135645891219275043ul>();
  return 0;
}
