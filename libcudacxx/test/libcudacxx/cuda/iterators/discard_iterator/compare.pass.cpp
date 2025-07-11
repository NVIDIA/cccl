//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// friend constexpr bool operator==(const discard_iterator& x, const discard_iterator& y);
// friend constexpr bool operator==(const discard_iterator& x, default_sentinel_t);

#include <cuda/iterator>
#include <cuda/std/utility>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  const int offset1 = 3;
  const int offset2 = 4;
  cuda::discard_iterator iter1(offset1);
  cuda::discard_iterator iter2(offset2);

  // equality
  assert(iter1 == iter1);
  assert(iter1 != iter2);

  assert(cuda::std::as_const(iter1) == iter1);
  assert(iter1 != cuda::std::as_const(iter2));

  assert(iter1 == cuda::std::as_const(iter1));
  assert(cuda::std::as_const(iter1) != iter2);

  assert(cuda::std::as_const(iter1) == cuda::std::as_const(iter1));
  assert(cuda::std::as_const(iter1) != cuda::std::as_const(iter2));

  // relation
  assert(iter1 < iter2);
  assert(iter1 <= iter2);
  assert(iter2 > iter1);
  assert(iter2 >= iter1);

  assert(cuda::std::as_const(iter1) < iter2);
  assert(cuda::std::as_const(iter1) <= iter2);
  assert(cuda::std::as_const(iter2) > iter1);
  assert(cuda::std::as_const(iter2) >= iter1);

  assert(iter1 < cuda::std::as_const(iter2));
  assert(iter1 <= cuda::std::as_const(iter2));
  assert(iter2 > cuda::std::as_const(iter1));
  assert(iter2 >= cuda::std::as_const(iter1));

  assert(cuda::std::as_const(iter1) < cuda::std::as_const(iter2));
  assert(cuda::std::as_const(iter1) <= cuda::std::as_const(iter2));
  assert(cuda::std::as_const(iter2) > cuda::std::as_const(iter1));
  assert(cuda::std::as_const(iter2) >= cuda::std::as_const(iter1));

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  using cuda::std::strong_ordering::equal;
  using cuda::std::strong_ordering::greater;
  using cuda::std::strong_ordering::less;

  assert(iter1 <=> iter1 == equal);
  assert(iter2 <=> iter1 == greater);
  assert(iter1 <=> iter2 == less);

  assert(cuda::std::as_const(iter1) <=> iter1 == equal);
  assert(cuda::std::as_const(iter2) <=> iter1 == greater);
  assert(cuda::std::as_const(iter1) <=> iter2 == less);

  assert(iter1 <=> cuda::std::as_const(iter1) == equal);
  assert(iter2 <=> cuda::std::as_const(iter1) == greater);
  assert(iter1 <=> cuda::std::as_const(iter2) == less);

  assert(cuda::std::as_const(iter1) <=> cuda::std::as_const(iter1) == equal);
  assert(cuda::std::as_const(iter2) <=> cuda::std::as_const(iter1) == greater);
  assert(cuda::std::as_const(iter1) <=> cuda::std::as_const(iter2) == less);
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
