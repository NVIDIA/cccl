//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr discard_iterator& operator--()
// constexpr discard_iterator operator--(int)

#include <cuda/iterator>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  const int index = 3;
  cuda::discard_iterator iter(index);
  assert(iter-- == cuda::discard_iterator(index + 0));
  assert(--iter == cuda::discard_iterator(index - 2));

  static_assert(cuda::std::is_same_v<decltype(iter--), cuda::discard_iterator>);
  static_assert(cuda::std::is_same_v<decltype(--iter), cuda::discard_iterator&>);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
