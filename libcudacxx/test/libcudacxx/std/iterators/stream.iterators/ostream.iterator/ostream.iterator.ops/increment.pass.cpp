//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// class ostream_iterator

// ostream_iterator& operator++();
// ostream_iterator& operator++(int);

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#  include <cuda/std/cassert>
#  include <cuda/std/sstream>

#  include "test_macros.h"

int main(int, char**)
{
  cuda::std::ostringstream os;
  cuda::std::ostream_iterator<int> i(os);
  cuda::std::ostream_iterator<int>& iref1 = ++i;
  assert(&iref1 == &i);
  cuda::std::ostream_iterator<int>& iref2 = i++;
  assert(&iref2 == &i);

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
