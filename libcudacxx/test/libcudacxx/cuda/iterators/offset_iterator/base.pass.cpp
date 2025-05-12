//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr const I& base() const &;
// constexpr I base() &&;

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    const int offset = 2;
    cuda::offset_iterator iter(random_access_iterator<int*>{buffer}, offset);
    assert(base(iter.base()) == buffer);
    assert(base(cuda::std::move(iter).base()) == buffer);

    static_assert(cuda::std::is_same_v<decltype(iter.base()), const random_access_iterator<int*>&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(iter).base()), random_access_iterator<int*>>);
  }

  {
    const int offset[] = {2};
    cuda::offset_iterator iter(random_access_iterator<int*>{buffer}, offset);
    assert(base(iter.base()) == buffer);
    assert(base(cuda::std::move(iter).base()) == buffer);

    static_assert(cuda::std::is_same_v<decltype(iter.base()), const random_access_iterator<int*>&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(iter).base()), random_access_iterator<int*>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
