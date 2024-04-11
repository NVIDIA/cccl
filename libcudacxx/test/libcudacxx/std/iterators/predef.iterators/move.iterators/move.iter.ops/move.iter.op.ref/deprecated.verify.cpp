//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// move_iterator

#include <cuda/std/iterator>

int main(int, char**)
{
  (void) cuda::std::move_iterator<int*>().operator->();
  // expected-warning@-1{{'operator->' is deprecated}}

  return 0;
}
