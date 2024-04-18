//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator InIter1, InputIterator InIter2, typename OutIter,
//          CopyConstructible Compare>
//   requires OutputIterator<OutIter, InIter1::reference>
//         && OutputIterator<OutIter, InIter2::reference>
//         && Predicate<Compare, InIter1::value_type, InIter2::value_type>
//         && Predicate<Compare, InIter2::value_type, InIter1::value_type>
//   OutIter
//   set_union(InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
//             OutIter result, Compare comp);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "MoveOnly.h"
#include "test_macros.h"

int main(int, char**)
{
  MoveOnly lhs[] = {MoveOnly(2)};
  MoveOnly rhs[] = {MoveOnly(2)};

  MoveOnly res[] = {MoveOnly(0)};
  cuda::std::set_union(
    cuda::std::make_move_iterator(lhs),
    cuda::std::make_move_iterator(lhs + 1),
    cuda::std::make_move_iterator(rhs),
    cuda::std::make_move_iterator(rhs + 1),
    res);

  assert(res[0].get() == 2);

  return 0;
}
