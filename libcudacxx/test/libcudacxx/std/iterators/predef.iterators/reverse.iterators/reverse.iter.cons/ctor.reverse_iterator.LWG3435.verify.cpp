//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// reverse_iterator

// template <class U>
//  requires !same_as<U, Iter> && convertible_to<const U&, Iter>
// reverse_iterator(const reverse_iterator<U> &);

#include <cuda/std/iterator>

struct Base
{};
struct Derived : Base
{};

__host__ __device__ void test()
{
  cuda::std::reverse_iterator<Base*> base;
  cuda::std::reverse_iterator<Derived*> derived(base); // expected-error {{no matching constructor for initialization of
                                                       // 'cuda::std::reverse_iterator<Derived *>'}}
}

int main(int, char**)
{
  return 0;
}
