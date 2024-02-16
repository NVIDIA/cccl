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
TEST_HOST_DEVICE
void
test()
{
    D d;
    assert(d.count() == typename D::rep());
    constexpr D d2 = D();
    static_assert(d2.count() == typename D::rep(), "");
}

int main(int, char**)
{
    test<cuda::std::chrono::duration<Rep> >();

  return 0;
}
