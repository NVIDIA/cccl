//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// static constexpr duration max(); // noexcept after C++17

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/limits>

#include "../../rep.h"
#include "test_macros.h"

template <class D>
__host__ __device__ void test()
{
  LIBCPP_ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<typename D::rep>::max());
#if TEST_STD_VER > 2017
  ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<typename D::rep>::max());
#endif
  {
    typedef typename D::rep Rep;
    Rep max_rep = cuda::std::chrono::duration_values<Rep>::max();
    assert(D::max().count() == max_rep);
  }
  {
    typedef typename D::rep Rep;
    constexpr Rep max_rep = cuda::std::chrono::duration_values<Rep>::max();
    static_assert(D::max().count() == max_rep, "");
  }
}

int main(int, char**)
{
  test<cuda::std::chrono::duration<int>>();
  test<cuda::std::chrono::duration<Rep>>();

  return 0;
}
