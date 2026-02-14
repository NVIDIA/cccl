//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// reverse_iterator

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class T>
_CCCL_CONCEPT HasMinus = _CCCL_REQUIRES_EXPR((T), T t)(unused(t - t));

using sized_it = random_access_iterator<int*>;
static_assert(cuda::std::sized_sentinel_for<sized_it, sized_it>);
static_assert(
  cuda::std::sized_sentinel_for<cuda::std::reverse_iterator<sized_it>, cuda::std::reverse_iterator<sized_it>>);
static_assert(HasMinus<cuda::std::reverse_iterator<sized_it>>);

// Check that `sized_sentinel_for` is false for `reverse_iterator`s if it is false for the underlying iterators.
using unsized_it = bidirectional_iterator<int*>;
static_assert(!cuda::std::sized_sentinel_for<unsized_it, unsized_it>);
static_assert(
  !cuda::std::sized_sentinel_for<cuda::std::reverse_iterator<unsized_it>, cuda::std::reverse_iterator<unsized_it>>);
static_assert(!HasMinus<cuda::std::reverse_iterator<unsized_it>>);

int main(int, char**)
{
  return 0;
}
