//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter>
//   max_element(Iter first, Iter last);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>

#include "test_iterators.h"

int main(int, char**)
{
  int arr[]    = {1, 2, 3};
  const int *b = cuda::std::begin(arr), *e = cuda::std::end(arr);
  using Iter = cpp17_input_iterator<const int*>;
  {
    // expected-error@*:* {{cuda::std::min_element requires a ForwardIterator}}
    (void) cuda::std::min_element(Iter(b), Iter(e));
  }
  {
    // expected-error@*:* {{cuda::std::max_element requires a ForwardIterator}}
    (void) cuda::std::max_element(Iter(b), Iter(e));
  }
  {
    // expected-error@*:* {{cuda::std::minmax_element requires a ForwardIterator}}
    (void) cuda::std::minmax_element(Iter(b), Iter(e));
  }

  return 0;
}
