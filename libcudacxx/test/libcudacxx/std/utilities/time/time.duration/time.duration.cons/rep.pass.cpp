//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// template <class Rep2>
//   explicit duration(const Rep2& r);

#include "../../rep.h"

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/ratio>

#include "test_macros.h"

struct NotValueConvertible
{
  __host__ __device__ operator int() const&& = delete;
  __host__ __device__ constexpr operator int() const&
  {
    return 1;
  }
};

template <class D, class R>
__host__ __device__ constexpr void check(R r)
{
  D d(r);
  assert(d.count() == r);
}

__host__ __device__ constexpr bool test()
{
  check<cuda::std::chrono::duration<int>>(5);
  check<cuda::std::chrono::duration<int, cuda::std::ratio<3, 2>>>(5);
  check<cuda::std::chrono::duration<Rep, cuda::std::ratio<3, 2>>>(Rep(3));
  check<cuda::std::chrono::duration<double, cuda::std::ratio<2, 3>>>(5.5);

  // test for [time.duration.cons]/1
  check<cuda::std::chrono::duration<int>>(NotValueConvertible());

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
