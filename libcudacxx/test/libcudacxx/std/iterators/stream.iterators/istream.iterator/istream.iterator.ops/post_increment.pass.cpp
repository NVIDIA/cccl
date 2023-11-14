//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// class istream_iterator

// istream_iterator operator++(int);

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#include <cuda/std/sstream>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    cuda::std::istringstream inf(" 1 23");
    cuda::std::istream_iterator<int> i(inf);
    cuda::std::istream_iterator<int> icopy = i++;
    assert(icopy == i);
    int j = 0;
    j = *i;
    assert(j == 23);
    j = 0;
    j = *icopy;
    assert(j == 1);

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
