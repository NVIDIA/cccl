//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/__simd_>

// [simd.iterator], ranges iterator/sentinel concept conformance for
// basic_vec::iterator, basic_vec::const_iterator, basic_mask::iterator,
// basic_mask::const_iterator against cuda::std::default_sentinel_t.

#include <cuda/std/__simd_>
#include <cuda/std/iterator>

#include "test_macros.h"

namespace simd = cuda::std::simd;

template <typename Iter>
TEST_FUNC void check_iterator_concepts()
{
  using Sentinel = cuda::std::default_sentinel_t;

  static_assert(cuda::std::input_iterator<Iter>);
  static_assert(cuda::std::forward_iterator<Iter>);
  static_assert(cuda::std::bidirectional_iterator<Iter>);
  static_assert(cuda::std::random_access_iterator<Iter>);
  static_assert(!cuda::std::contiguous_iterator<Iter>);
  static_assert(cuda::std::sentinel_for<Iter, Iter>);
  static_assert(cuda::std::sentinel_for<Sentinel, Iter>);
  static_assert(cuda::std::sized_sentinel_for<Iter, Iter>);
  static_assert(cuda::std::sized_sentinel_for<Sentinel, Iter>);
}

TEST_FUNC void test()
{
  using Vec  = simd::basic_vec<int, simd::fixed_size<4>>;
  using Mask = typename Vec::mask_type;

  check_iterator_concepts<typename Vec::iterator>();
  check_iterator_concepts<typename Vec::const_iterator>();
  check_iterator_concepts<typename Mask::iterator>();
  check_iterator_concepts<typename Mask::const_iterator>();
}

int main(int, char**)
{
  return 0;
}
