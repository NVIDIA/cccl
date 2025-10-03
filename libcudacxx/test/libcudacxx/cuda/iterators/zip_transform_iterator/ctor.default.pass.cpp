//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// zip_transform_iterator() = default;

#include <cuda/iterator>
#include <cuda/std/tuple>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  {
    cuda::zip_transform_iterator<Plus, PODIter, PODIter> iter;
    assert(*iter == 0); // PODIter has to be initialised to have value 0
  }

  {
    cuda::zip_transform_iterator<Plus, PODIter, PODIter> iter{};
    assert(*iter == 0); // PODIter has to be initialised to have value 0
  }

  // All iterators need to be default constructible
  static_assert(
    !cuda::std::is_default_constructible_v<cuda::zip_transform_iterator<Plus, PODIter, IterNotDefaultConstructible>>);

  // The functor needs to be default constructible
  static_assert(
    !cuda::std::is_default_constructible_v<cuda::zip_transform_iterator<PlusNotDefaultConstructible, PODIter, PODIter>>);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
