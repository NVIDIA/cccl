//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// UNSUPPORTED: nvrtc

// reverse_iterator

// template <class U>
//   requires HasAssign<Iter, const U&>
//   reverse_iterator&
//   operator=(const reverse_iterator<U>& u);

// test requires

#include <cuda/std/iterator>

template <class It, class U>
__host__ __device__
void
test(U u)
{
    const cuda::std::reverse_iterator<U> r2(u);
    cuda::std::reverse_iterator<It> r1;
    r1 = r2;
}

struct base {};
struct derived {};

int main(int, char**)
{
    derived d;
    test<base*>(&d);

  return 0;
}
