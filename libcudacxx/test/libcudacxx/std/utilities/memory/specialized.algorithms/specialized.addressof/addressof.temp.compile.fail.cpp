//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11

// <memory>

// template <ObjectType T> T* addressof(T&& r) = delete;

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  const int* p = cuda::std::addressof<const int>(0);

  return 0;
}
