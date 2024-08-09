//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: msvc-19.16

// class cuda::std::ranges::subrange;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

static_assert(cuda::std::is_convertible_v<ForwardSubrange, cuda::std::pair<ForwardIter, ForwardIter>>);
static_assert(cuda::std::is_convertible_v<ForwardSubrange, cuda::std::tuple<ForwardIter, ForwardIter>>);
static_assert(!cuda::std::is_convertible_v<ForwardSubrange, cuda::std::tuple<ForwardIter, ForwardIter>&>);
static_assert(!cuda::std::is_convertible_v<ForwardSubrange, cuda::std::tuple<ForwardIter, ForwardIter, ForwardIter>>);
static_assert(cuda::std::is_convertible_v<ConvertibleForwardSubrange, cuda::std::tuple<ConvertibleForwardIter, int*>>);
static_assert(!cuda::std::is_convertible_v<SizedIntPtrSubrange, cuda::std::tuple<long*, int*>>);
static_assert(cuda::std::is_convertible_v<SizedIntPtrSubrange, cuda::std::tuple<int*, int*>>);

__host__ __device__ constexpr bool test()
{
  ForwardSubrange a(ForwardIter(globalBuff), ForwardIter(globalBuff + 8));
  cuda::std::pair<ForwardIter, ForwardIter> aPair = a;
  assert(base(aPair.first) == globalBuff);
  assert(base(aPair.second) == globalBuff + 8);
  cuda::std::tuple<ForwardIter, ForwardIter> aTuple = a;
  assert(base(cuda::std::get<0>(aTuple)) == globalBuff);
  assert(base(cuda::std::get<1>(aTuple)) == globalBuff + 8);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
