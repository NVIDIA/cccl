//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// move_iterator

// explicit move_iterator(Iter );

// test explicit

#include <cuda/std/iterator>

template <class It>
__host__ __device__
void
test(It i)
{
    cuda::std::move_iterator<It> r = i;
}

int main(int, char**)
{
    char s[] = "123";
    test(s);

  return 0;
}
