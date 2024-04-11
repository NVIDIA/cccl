//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// template<class I, class R = ranges::less, class P = identity>
//   concept sortable = see below;                            // since C++20

#include <cuda/std/functional>
#include <cuda/std/iterator>

using CompInt     = bool (*)(int, int);
using CompDefault = cuda::std::ranges::less;

using AllConstraintsSatisfied = int*;
static_assert(cuda::std::permutable<AllConstraintsSatisfied>);
static_assert(cuda::std::indirect_strict_weak_order<CompDefault, AllConstraintsSatisfied>);
static_assert(cuda::std::sortable<AllConstraintsSatisfied>);
static_assert(cuda::std::indirect_strict_weak_order<CompInt, AllConstraintsSatisfied>);
static_assert(cuda::std::sortable<AllConstraintsSatisfied, CompInt>);

struct Foo
{};
using Proj = int (*)(Foo);
static_assert(cuda::std::permutable<Foo*>);
static_assert(!cuda::std::indirect_strict_weak_order<CompDefault, Foo*>);
static_assert(cuda::std::indirect_strict_weak_order<CompDefault, cuda::std::projected<Foo*, Proj>>);
static_assert(!cuda::std::sortable<Foo*, CompDefault>);
static_assert(cuda::std::sortable<Foo*, CompDefault, Proj>);
static_assert(!cuda::std::indirect_strict_weak_order<CompInt, Foo*>);
static_assert(cuda::std::indirect_strict_weak_order<CompInt, cuda::std::projected<Foo*, Proj>>);
static_assert(!cuda::std::sortable<Foo*, CompInt>);
static_assert(cuda::std::sortable<Foo*, CompInt, Proj>);

using NotPermutable = const int*;
static_assert(!cuda::std::permutable<NotPermutable>);
static_assert(cuda::std::indirect_strict_weak_order<CompInt, NotPermutable>);
static_assert(!cuda::std::sortable<NotPermutable, CompInt>);

struct Empty
{};
using NoIndirectStrictWeakOrder = Empty*;
static_assert(cuda::std::permutable<NoIndirectStrictWeakOrder>);
static_assert(!cuda::std::indirect_strict_weak_order<CompInt, NoIndirectStrictWeakOrder>);
static_assert(!cuda::std::sortable<NoIndirectStrictWeakOrder, CompInt>);

int main(int, char**)
{
  return 0;
}
