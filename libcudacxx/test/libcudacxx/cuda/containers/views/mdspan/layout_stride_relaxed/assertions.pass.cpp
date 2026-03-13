//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// ADDITIONAL_COMPILE_DEFINITIONS: _CCCL_ENABLE_ASSERTIONS

// <cuda/mdspan>

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "check_assertion.h"
#include "test_macros.h"

void test_negative_offset_assertion()
{
  using extents_t = cuda::std::extents<int, 3>;
  using mapping_t = cuda::layout_stride_relaxed::mapping<extents_t>;
  using strides_t = typename mapping_t::strides_type;
  using offset_t  = typename mapping_t::offset_type;

  TEST_CCCL_ASSERT_FAILURE(mapping_t(extents_t{}, strides_t(1), static_cast<offset_t>(-1)),
                           "layout_stride_relaxed::mapping: offset must be nonnegative");
}

void test_static_stride_comparison()
{
  using extents_t = cuda::std::extents<int, 2, 3>;
  using right_map = cuda::std::layout_right::mapping<extents_t>;

  right_map rhs(extents_t{});

  using good_strides = cuda::strides<cuda::std::ptrdiff_t, 3, 1>;
  using good_map     = cuda::layout_stride_relaxed::mapping<extents_t, good_strides>;
  good_map lhs_good(extents_t{}, good_strides{});
  assert(lhs_good == rhs);

  using bad_strides = cuda::strides<cuda::std::ptrdiff_t, 4, 1>;
  using bad_map     = cuda::layout_stride_relaxed::mapping<extents_t, bad_strides>;
  bad_map lhs_bad(extents_t{}, bad_strides{});
  TEST_CCCL_ASSERT_FAILURE((lhs_bad == rhs),
                           "strides construction: mismatch of provided arguments with static strides.");
}

void test_strides_narrowing_assertion()
{
  using big_strides   = cuda::strides<cuda::std::int64_t, cuda::dynamic_stride>;
  using small_strides = cuda::strides<cuda::std::int8_t, cuda::dynamic_stride>;

  big_strides ok_big(100);
  small_strides ok_small = ok_big;
  assert(ok_small.stride(0) == 100);

  big_strides too_big(200);
  TEST_CCCL_ASSERT_FAILURE((small_strides(too_big)), "strides construction: stride is out of range");
}

int main(int, char**)
{
  test_negative_offset_assertion();
  test_static_stride_comparison();
  test_strides_narrowing_assertion();
  return 0;
}
