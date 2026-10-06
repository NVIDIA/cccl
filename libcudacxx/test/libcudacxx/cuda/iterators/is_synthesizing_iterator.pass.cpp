//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/iterator>

#include "test_macros.h"

struct Fn
{};

template <class Iter, bool Expected>
TEST_FUNC constexpr void test()
{
  static_assert(cuda::__is_synthesizing_iterator_v<Iter> == Expected);
  static_assert(cuda::__is_synthesizing_iterator_v<const Iter> == Expected);
  static_assert(cuda::__is_synthesizing_iterator_v<volatile Iter> == Expected);
  static_assert(cuda::__is_synthesizing_iterator_v<const volatile Iter> == Expected);
  static_assert(cuda::__is_synthesizing_iterator_v<Iter&> == Expected);
  static_assert(cuda::__is_synthesizing_iterator_v<const Iter&> == Expected);
  static_assert(cuda::__is_synthesizing_iterator_v<Iter&&> == Expected);
}

TEST_FUNC constexpr bool test()
{
  using Counting  = cuda::counting_iterator<int>;
  using Constant  = cuda::constant_iterator<int>;
  using Shuffle   = cuda::shuffle_iterator<>;
  using Strided   = cuda::strided_iterator<Counting, int>;
  using Transform = cuda::transform_iterator<Fn, Strided>;
  using Zip       = cuda::zip_iterator<Transform, Constant>;
  using ZipFn     = cuda::zip_transform_iterator<Fn, Counting, Constant>;

  test<int*, false>();
  test<Counting, true>();
  test<Constant, true>();
  test<Shuffle, true>();
  test<cuda::discard_iterator, false>();
  test<cuda::permutation_iterator<int*, Counting>, false>();
  test<cuda::tabulate_output_iterator<Fn>, false>();
  test<cuda::transform_output_iterator<Fn, int*>, false>();
  test<cuda::transform_input_output_iterator<Fn, Fn, int*>, false>();

  test<Strided, true>();
  test<cuda::strided_iterator<int*, int>, false>();
  test<Transform, true>();
  test<cuda::transform_iterator<Fn, int*>, false>();
  test<cuda::transform_iterator<Fn, cuda::transform_iterator<Fn, Counting>>, true>();

  test<Zip, true>();
  test<cuda::zip_iterator<Counting, int*>, false>();
  test<cuda::zip_iterator<Counting, Constant>, true>();
  test<cuda::strided_iterator<cuda::zip_iterator<Counting, int*>, int>, false>();

  test<ZipFn, true>();
  test<cuda::zip_transform_iterator<Fn, Counting, int*>, false>();
  test<cuda::zip_transform_iterator<Fn, Transform, Zip>, true>();
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
