//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr explicit iterator(W value);

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
#if TEST_STD_VER >= 2020
  { // CTAD
    const int val = 42;
    cuda::counting_iterator iter{val};
    assert(*iter == 42);
  }

  { // CTAD
    cuda::counting_iterator iter{42};
    assert(*iter == 42);
  }
#endif // TEST_STD_VER >= 2020

  {
    const int val = 42;
    cuda::counting_iterator<int> iter{val};
    assert(*iter == 42);
  }

  {
    cuda::counting_iterator<int> iter{42};
    assert(*iter == 42);
  }

  {
    const Int42<ValueCtor> val{42};
    cuda::counting_iterator<Int42<ValueCtor>> iter{val};
    assert(*iter == Int42<ValueCtor>{42});
  }

  {
    cuda::counting_iterator<Int42<ValueCtor>> iter{Int42<ValueCtor>{42}};
    assert(*iter == Int42<ValueCtor>{42});
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
