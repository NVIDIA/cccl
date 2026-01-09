//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr reference operator*() noexcept;
// constexpr const_reference operator*() const noexcept;

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

template <class Stride>
__host__ __device__ constexpr void test(Stride stride)
{
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};
  {
    cuda::strided_iterator iter{buffer, stride};
    assert(*iter == 1);
    assert(cuda::std::addressof(*iter) == buffer);

    ++iter;
    assert(*iter == 3);
    *iter = 5;
    assert(buffer[iter.stride()] == 5);
    assert(cuda::std::addressof(*iter) == buffer + iter.stride());
    static_assert(noexcept(*iter));
    static_assert(cuda::std::is_same_v<decltype(*iter), cuda::std::iter_reference_t<int*>>);
  }

  {
    const cuda::strided_iterator citer{buffer, stride};
    assert(*citer == 1);
    assert(cuda::std::addressof(*citer) == buffer);

    static_assert(noexcept(*citer));
    static_assert(cuda::std::is_same_v<decltype(*citer), cuda::std::iter_reference_t<int*>>);
  }
}

__host__ __device__ constexpr bool test()
{
  test(2);
  test(Stride<2>{});

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
