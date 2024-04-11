//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <functional>

// reference_wrapper

// reference_wrapper(T&&) = delete;

// #include <cuda/std/functional>
#include <cuda/std/cassert>
#include <cuda/std/utility>

int main(int, char**)
{
  cuda::std::reference_wrapper<const int> r(3);

  return 0;
}
