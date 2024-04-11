//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// const T* data() const;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstddef> // for cuda::std::max_align_t
#include <cuda/std/cstdint>

#include "test_macros.h"

struct NoDefault
{
  __host__ __device__ TEST_CONSTEXPR NoDefault(int) {}
};

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    const C c = {1, 2, 3.5};
    ASSERT_NOEXCEPT(c.data());
    const T* p = c.data();
    assert(p[0] == 1);
    assert(p[1] == 2);
    assert(p[2] == 3.5);
  }
  {
    typedef double T;
    typedef cuda::std::array<T, 0> C;
    const C c = {};
    ASSERT_NOEXCEPT(c.data());
    const T* p = c.data();
    unused(p);
  }
  {
    typedef NoDefault T;
    typedef cuda::std::array<T, 0> C;
    const C c = {};
    ASSERT_NOEXCEPT(c.data());
    const T* p = c.data();
    unused(p);
  }
  {
    cuda::std::array<int, 5> const c = {0, 1, 2, 3, 4};
    assert(c.data() == &c[0]);
    assert(*c.data() == c[0]);
  }

  return true;
}

int main(int, char**)
{
  tests();
#if TEST_STD_VER >= 2014
  static_assert(tests(), "");
#endif

  // Test the alignment of data()
  {
    typedef cuda::std::max_align_t T;
    typedef cuda::std::array<T, 0> C;
    const C c                 = {};
    const T* p                = c.data();
    cuda::std::uintptr_t pint = reinterpret_cast<cuda::std::uintptr_t>(p);
    assert(pint % TEST_ALIGNOF(T) == 0);
  }

  return 0;
}
