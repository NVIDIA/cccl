//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class T>
// using iter_reference_t = decltype(*declval<T&>());

#include <cuda/std/concepts>
#include <cuda/std/iterator>

#include "test_iterators.h"

static_assert(cuda::std::same_as<cuda::std::iter_reference_t<cpp17_input_iterator<int*>>, int&>);
static_assert(cuda::std::same_as<cuda::std::iter_reference_t<forward_iterator<int*>>, int&>);
static_assert(cuda::std::same_as<cuda::std::iter_reference_t<bidirectional_iterator<int*>>, int&>);
static_assert(cuda::std::same_as<cuda::std::iter_reference_t<random_access_iterator<int*>>, int&>);

int main(int, char**)
{
  return 0;
}
