//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// <memory>

// unique_ptr

// test op*()

#include <cuda/std/__memory_>

#include "test_macros.h"

void f()
{
  cuda::std::unique_ptr<int[]> p(new int(3));
  const cuda::std::unique_ptr<int[]>& cp = p;
  TEST_IGNORE_NODISCARD(*p); // expected-error-re {{indirection requires pointer operand ('cuda::std::unique_ptr<int{{[
                             // ]*}}[]>' invalid)}}
  TEST_IGNORE_NODISCARD(*cp); // expected-error-re {{indirection requires pointer operand ('const
                              // cuda::std::unique_ptr<int{{[ ]*}}[]>' invalid)}}
}
