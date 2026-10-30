//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// P3323R1: cv-qualified types in atomic and atomic_ref
// atomic<cv T> is ill-formed.

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// UNSUPPORTED: force-tile

#include <cuda/std/atomic>

int main(int, char**)
{
  // P3323R1: atomic<const T> is ill-formed because T must be cv-unqualified.
  cuda::std::atomic<const int> a(0); // expected-error@*:* {{requires T to be cv-unqualified}}
  (void) a;

  return 0;
}
