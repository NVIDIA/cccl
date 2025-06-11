//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr W operator*() const noexcept;

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

template <class T>
__host__ __device__ constexpr void test(T value)
{
  {
    cuda::constant_iterator iter{value, 42};
    for (int i = 0; i < 100; ++i, ++iter)
    {
      assert(*iter == value);
    }

    static_assert(noexcept(*iter));
    static_assert(cuda::std::is_same_v<decltype(*iter), const T&>);
  }

  {
    const cuda::constant_iterator iter{value, 42};
    assert(*iter == value);
    static_assert(noexcept(*iter));
    static_assert(cuda::std::is_same_v<decltype(*iter), const T&>);
  }
}

__host__ __device__ constexpr bool test()
{
  test(42);
  test(NotDefaultConstructible{42});

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
