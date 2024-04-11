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

// const T* operator->() const;

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#  include <cuda/std/cassert>
#  include <cuda/std/sstream>

#  include "test_macros.h"

struct A
{
  double d_;
  int i_;
};

void operator&(A const&) {}

cuda::std::istream& operator>>(cuda::std::istream& is, A& a)
{
  return is >> a.d_ >> a.i_;
}

int main(int, char**)
{
  cuda::std::istringstream inf("1.5  23 ");
  cuda::std::istream_iterator<A> i(inf);
  assert(i->d_ == 1.5);
  assert(i->i_ == 23);

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
