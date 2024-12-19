//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// If we have a copy-propagating cache, when we copy ZeroOnDestroy, we will get a
// dangling reference to the copied-from object. This test ensures that we do not
// propagate the cache on copy.

#include <cuda/std/cstddef>
#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

struct ZeroOnDestroy : cuda::std::ranges::view_base
{
  unsigned count = 0;
  int buff[8]    = {1, 2, 3, 4, 5, 6, 7, 8};

  __host__ __device__ constexpr ForwardIter begin()
  {
    return ForwardIter(buff);
  }
  __host__ __device__ constexpr ForwardIter begin() const
  {
    return ForwardIter();
  }
  __host__ __device__ constexpr ForwardIter end()
  {
    return ForwardIter(buff + 8);
  }
  __host__ __device__ constexpr ForwardIter end() const
  {
    return ForwardIter();
  }

  __host__ __device__ ~ZeroOnDestroy()
  {
    for (unsigned i = 0; i < count; ++i)
    {
      buff[i] = 0;
    }
  }

  __host__ __device__ static auto dropFirstFour()
  {
    ZeroOnDestroy zod;
    cuda::std::ranges::drop_view dv(zod, 4);
    // Make sure we call begin here so the next call to begin will
    // use the cached iterator.
    assert(*dv.begin() == 5);
    // Intentionally invoke the copy ctor here.
    return cuda::std::ranges::drop_view(dv);
  }
};

int main(int, char**)
{
  auto noDanlingCache = ZeroOnDestroy::dropFirstFour();
  // If we use the cached version, it will reference the copied-from view.
  // Worst case this is a segfault, best case it's an assertion fired.
  assert(*noDanlingCache.begin() == 5);

  return 0;
}
