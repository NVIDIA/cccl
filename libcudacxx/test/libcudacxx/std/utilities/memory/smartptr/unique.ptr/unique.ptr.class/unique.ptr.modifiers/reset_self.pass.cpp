//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: dynamic allocation is not supported in tile mode

// A owns itself, so ptxas sees a recursive destructor even though reset clears ptr_ before deleting A.
// ADDITIONAL_COMPILE_OPTIONS_PTXAS: --suppress-stack-size-warning

// <memory>

// unique_ptr

// test reset against resetting self

#include <cuda/std/__memory_>

#include "test_macros.h"

struct A
{
  cuda::std::unique_ptr<A> ptr_;
};

TEST_HOST_DEVICE_FUNC TEST_CONSTEXPR_CXX23 bool test()
{
  auto* self = new A;
  self->ptr_.reset(self);
  self->ptr_.reset();

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2023
  static_assert(test());
#endif // TEST_STD_VER >= 2023

  return 0;
}
