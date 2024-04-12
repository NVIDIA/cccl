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

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#  include <cuda/std/vector>

int main(int, char**)
{
  cuda::std::back_insert_iterator<cuda::std::vector<int>> i = cuda::std::vector<int>();

  return 0;
}
#else
int main(int, char**)
{
  assert();
  return 0;
}
#endif
