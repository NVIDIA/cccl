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

// istream_iterator(const istream_iterator& x);
//  C++17 says:  If is_trivially_copy_constructible_v<T> is true, then
//     this constructor is a trivial copy constructor.

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#  include <cuda/std/cassert>
#  include <cuda/std/sstream>

#  include "test_macros.h"

int main(int, char**)
{
  {
    cuda::std::istream_iterator<int> io;
    cuda::std::istream_iterator<int> i = io;
    assert(i == cuda::std::istream_iterator<int>());
  }
  {
    cuda::std::istringstream inf(" 1 23");
    cuda::std::istream_iterator<int> io(inf);
    cuda::std::istream_iterator<int> i = io;
    assert(i != cuda::std::istream_iterator<int>());
    int j = 0;
    j     = *i;
    assert(j == 1);
  }

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
