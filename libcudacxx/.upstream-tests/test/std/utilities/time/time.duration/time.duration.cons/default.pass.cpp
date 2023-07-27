//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// duration() = default;

// Rep must be default initialized, not initialized with 0

#include <cuda/std/chrono>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../../rep.h"

template <class D>
__host__ __device__
void
test()
{
    D d;
    assert(d.count() == typename D::rep());
#if TEST_STD_VER >= 11
    constexpr D d2 = D();
    static_assert(d2.count() == typename D::rep(), "");
#endif
}

int main(int, char**)
{
    test<cuda::std::chrono::duration<Rep> >();

  return 0;
}
