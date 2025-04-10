//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_OPTIONS_HOST: -Wno-ctad-maybe-unsupported

// <mdspan>

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "test_macros.h"

// mdspan

// layout_stride::mapping does not have explicit deduction guides,
// but implicit deduction guides for constructor taking extents and strides
// should work

__host__ __device__ constexpr bool test()
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;

  static_assert(_CCCL_TRAIT(cuda::std::is_convertible, const unsigned&, int)
                && _CCCL_TRAIT(cuda::std::is_nothrow_constructible, int, const unsigned&));

  static_assert(cuda::std::is_same_v<
                decltype(cuda::std::layout_stride::mapping(cuda::std::extents<int>(), cuda::std::array<unsigned, 0>())),
                cuda::std::layout_stride::template mapping<cuda::std::extents<int>>>);
  static_assert(cuda::std::is_same_v<
                decltype(cuda::std::layout_stride::mapping(cuda::std::extents<int, 4>(), cuda::std::array<char, 1>{1})),
                cuda::std::layout_stride::template mapping<cuda::std::extents<int, 4>>>);
  static_assert(cuda::std::is_same_v<
                decltype(cuda::std::layout_stride::mapping(cuda::std::extents<int, D>(), cuda::std::array<char, 1>{1})),
                cuda::std::layout_stride::template mapping<cuda::std::extents<int, D>>>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::layout_stride::mapping(
                                       cuda::std::extents<unsigned, D, 3>(), cuda::std::array<int64_t, 2>{3, 100})),
                                     cuda::std::layout_stride::template mapping<cuda::std::extents<unsigned, D, 3>>>);

  static_assert(cuda::std::is_same_v<
                decltype(cuda::std::layout_stride::mapping(cuda::std::extents<int>(), cuda::std::span<unsigned, 0>())),
                cuda::std::layout_stride::template mapping<cuda::std::extents<int>>>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::layout_stride::mapping(
                                       cuda::std::extents<int, 4>(), cuda::std::declval<cuda::std::span<char, 1>>())),
                                     cuda::std::layout_stride::template mapping<cuda::std::extents<int, 4>>>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::layout_stride::mapping(
                                       cuda::std::extents<int, D>(), cuda::std::declval<cuda::std::span<char, 1>>())),
                                     cuda::std::layout_stride::template mapping<cuda::std::extents<int, D>>>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::std::layout_stride::mapping(
                           cuda::std::extents<unsigned, D, 3>(), cuda::std::declval<cuda::std::span<int64_t, 2>>())),
                         cuda::std::layout_stride::template mapping<cuda::std::extents<unsigned, D, 3>>>);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
