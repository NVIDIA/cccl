//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: true

// <algorithm>

// template<RandomAccessIterator Iter, StrictWeakOrder<auto, Iter::value_type> Compare>
//   requires ShuffleIterator<Iter>
//         && CopyConstructible<Compare>
//   void
//   sort(Iter first, Iter last, Compare comp);

#include <cuda/std/__memory_>
#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <cuda/std/numeric>

#include "test_macros.h"

struct indirect_less
{
  template <class P>
  __host__ __device__ bool operator()(const P& x, const P& y) const noexcept
  {
    return *x < *y;
  }
};

int main(int, char**)
{
  {
    cuda::std::array<int, 1000> v;
    cuda::std::iota(v.begin(), v.end(), 0);
    cuda::std::sort(v.begin(), v.end(), cuda::std::greater<int>());
    cuda::std::reverse(v.begin(), v.end());
    assert(cuda::std::is_sorted(v.begin(), v.end()));
  }

  {
    cuda::std::array<cuda::std::unique_ptr<int>, 1000> v;
    for (int i = 0; static_cast<cuda::std::size_t>(i) < v.size(); ++i)
    {
      v[i].reset(new int(i));
    }
    cuda::std::sort(v.begin(), v.end(), indirect_less());
    assert(cuda::std::is_sorted(v.begin(), v.end(), indirect_less{}));
    assert(*v[0] == 0);
    assert(*v[1] == 1);
    assert(*v[2] == 2);
  }

  return 0;
}
