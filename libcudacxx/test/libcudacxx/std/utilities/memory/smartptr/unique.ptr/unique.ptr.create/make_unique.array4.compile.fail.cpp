//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/string>

// UNSUPPORTED: nvrtc

int main(int, char**)
{
  auto up4 = cuda::std::make_unique<int[5]>(11, 22, 33, 44, 55); // deleted

  return 0;
}
