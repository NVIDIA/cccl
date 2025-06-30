//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/iterator>

#include "test_macros.h"
#include "types.h"

__host__ __device__ void test()
{
  using IterTraits = cuda::std::iterator_traits<cuda::transform_output_iterator<int*, PlusOne>>;

  static_assert(cuda::std::same_as<IterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
  static_assert(cuda::std::same_as<IterTraits::difference_type, cuda::std::ptrdiff_t>);
  static_assert(cuda::std::same_as<IterTraits::value_type, void>);
  static_assert(cuda::std::same_as<IterTraits::pointer, void>);
  static_assert(cuda::std::same_as<IterTraits::reference, void>);
  static_assert(cuda::std::input_or_output_iterator<cuda::transform_output_iterator<int*, PlusOne>>);
  static_assert(cuda::std::output_iterator<cuda::transform_output_iterator<int*, PlusOne>, int>);
}

int main(int, char**)
{
  return 0;
}
