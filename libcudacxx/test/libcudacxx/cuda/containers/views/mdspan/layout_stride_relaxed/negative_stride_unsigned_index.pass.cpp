//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// <cuda/__mdspan/layout_stride_relaxed.h>

#include <cuda/mdspan>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using E = cuda::std::extents<unsigned, 3>;
  using S = cuda::dstrides<cuda::std::int16_t, 1>;
  using M = cuda::layout_stride_relaxed::mapping<E, S>;

  using index_type  = typename M::index_type;
  using offset_type = typename M::offset_type;
  static_assert(cuda::std::is_unsigned_v<index_type>);
  static_assert(cuda::std::numeric_limits<index_type>::max() <= cuda::std::numeric_limits<int64_t>::max());

  constexpr offset_type offset = 10;
  M mapping(E{}, S(-2), offset);

  for (index_type i = 0; i < 3; ++i)
  {
    const auto expected = int64_t{offset} + static_cast<int64_t>(i) * -2;
    const auto got      = static_cast<int64_t>(mapping(i));
    assert(got == expected);
  }
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
