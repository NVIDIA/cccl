//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// back_insert_iterator

// explicit back_insert_iterator(Cont& x);

// test for explicit

#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>

int main(int, char**)
{
  cuda::std::back_insert_iterator<cuda::std::inplace_vector<int, 3>> i = cuda::std::inplace_vector<int, 3>();

  return 0;
}
