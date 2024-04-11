//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// static constexpr duration zero(); // noexcept after C++17

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "../../rep.h"
#include "test_macros.h"

template <class D>
__host__ __device__ void test()
{
  LIBCPP_ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<typename D::rep>::zero());
#if TEST_STD_VER > 2017
  ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<typename D::rep>::zero());
#endif
  {
    typedef typename D::rep Rep;
    Rep zero_rep = cuda::std::chrono::duration_values<Rep>::zero();
    assert(D::zero().count() == zero_rep);
  }
  {
    typedef typename D::rep Rep;
    constexpr Rep zero_rep = cuda::std::chrono::duration_values<Rep>::zero();
    static_assert(D::zero().count() == zero_rep, "");
  }
}

int main(int, char**)
{
  test<cuda::std::chrono::duration<int>>();
  test<cuda::std::chrono::duration<Rep>>();

  return 0;
}
