//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// reverse_iterator

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

#if TEST_STD_VER > 2017
template <class T>
concept HasMinus = requires(T t) { t - t; };
#else
template <class T>
_LIBCUDACXX_CONCEPT_FRAGMENT(HasMinus_, requires(T t)(t - t));

template <class T>
_LIBCUDACXX_CONCEPT HasMinus = _LIBCUDACXX_FRAGMENT(HasMinus_, T);
#endif

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
