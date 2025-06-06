//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// iterator() requires default_initializable<Fn> = default;

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  {
    cuda::tabulate_output_iterator<basic_functor> iter;
    assert(iter.index() == 0);
    *iter = 0;
    static_assert(
      cuda::std::is_same_v<decltype(iter), cuda::tabulate_output_iterator<basic_functor, cuda::std::ptrdiff_t>>);
  }

  {
    const cuda::tabulate_output_iterator<basic_functor> iter;
    assert(iter.index() == 0);
    *iter = 0;
    static_assert(
      cuda::std::is_same_v<decltype(iter), const cuda::tabulate_output_iterator<basic_functor, cuda::std::ptrdiff_t>>);
  }

  {
    cuda::tabulate_output_iterator<mutable_functor> iter;
    assert(iter.index() == 0);
    *iter = 0;
    static_assert(
      cuda::std::is_same_v<decltype(iter), cuda::tabulate_output_iterator<mutable_functor, cuda::std::ptrdiff_t>>);
  }

  {
    static_assert(
      !cuda::std::is_default_constructible_v<cuda::tabulate_output_iterator<not_default_constructible_functor>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
