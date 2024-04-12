//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

#include <cuda/std/iterator>

#include "test_macros.h"

struct Incomplete;
template <class T>
struct Holder
{
  T t;
};

template <class>
struct Intable
{
  __host__ __device__ operator int() const
  {
    return 1;
  }
};

int main(int, char**)
{
  Holder<Incomplete>* a[2] = {};
  Holder<Incomplete>** p   = a;
#if TEST_STD_VER > 2011
  p = cuda::std::next(p);
  p = cuda::std::prev(p);
  p = cuda::std::next(p, 2);
  p = cuda::std::prev(p, 2);
#endif
  cuda::std::advance(p, Intable<Holder<Incomplete>>());
  (void) cuda::std::distance(p, p);

  return 0;
}
