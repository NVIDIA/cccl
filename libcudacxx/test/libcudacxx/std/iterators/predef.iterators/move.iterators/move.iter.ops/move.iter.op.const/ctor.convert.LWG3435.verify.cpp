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

// template <class U>
//  requires !same_as<U, Iter> && convertible_to<const U&, Iter>
// move_iterator(const move_iterator<U> &u);

#include <cuda/std/iterator>

struct Base
{};
struct Derived : Base
{};

void test()
{
  cuda::std::move_iterator<Base*> base;
  cuda::std::move_iterator<Derived*> derived(base); // expected-error {{no matching constructor for initialization of
                                                    // 'std::move_iterator<Derived *>'}}
}
