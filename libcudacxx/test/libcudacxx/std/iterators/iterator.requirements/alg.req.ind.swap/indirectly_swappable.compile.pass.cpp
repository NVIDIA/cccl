//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// template<class I1, class I2>
// concept indirectly_swappable;

#include <cuda/std/iterator>

#include "test_macros.h"

template <class T, class ValueType = T>
struct PointerTo
{
  using value_type = ValueType;
  TEST_FUNC T& operator*() const;
};

static_assert(cuda::std::indirectly_swappable<PointerTo<int>>);
static_assert(cuda::std::indirectly_swappable<PointerTo<int>, PointerTo<int>>);

struct B;

struct A
{
  TEST_FUNC friend void iter_swap(const PointerTo<A>&, const PointerTo<A>&);
};

// Is indirectly swappable.
struct B
{
  TEST_FUNC friend void iter_swap(const PointerTo<B>&, const PointerTo<B>&);
  TEST_FUNC friend void iter_swap(const PointerTo<A>&, const PointerTo<B>&);
  TEST_FUNC friend void iter_swap(const PointerTo<B>&, const PointerTo<A>&);
};

// Valid except ranges::iter_swap(i2, i1).
struct C
{
  TEST_FUNC friend void iter_swap(const PointerTo<C>&, const PointerTo<C>&);
  TEST_FUNC friend void iter_swap(const PointerTo<A>&, const PointerTo<C>&);
  TEST_FUNC friend void iter_swap(const PointerTo<C>&, const PointerTo<A>&) = delete;
};

// Valid except ranges::iter_swap(i1, i2).
struct D
{
  TEST_FUNC friend void iter_swap(const PointerTo<D>&, const PointerTo<D>&);
  TEST_FUNC friend void iter_swap(const PointerTo<A>&, const PointerTo<D>&) = delete;
  TEST_FUNC friend void iter_swap(const PointerTo<D>&, const PointerTo<A>&);
};

// Valid except ranges::iter_swap(i2, i2).
struct E
{
  E operator=(const E&)                                                     = delete;
  TEST_FUNC friend void iter_swap(const PointerTo<E>&, const PointerTo<E>&) = delete;
  TEST_FUNC friend void iter_swap(const PointerTo<A>&, const PointerTo<E>&);
  TEST_FUNC friend void iter_swap(const PointerTo<E>&, const PointerTo<A>&);
};

struct F
{
  TEST_FUNC friend void iter_swap(const PointerTo<F>&, const PointerTo<F>&) = delete;
};

// Valid except ranges::iter_swap(i1, i1).
struct G
{
  TEST_FUNC friend void iter_swap(const PointerTo<G>&, const PointerTo<G>&);
  TEST_FUNC friend void iter_swap(const PointerTo<F>&, const PointerTo<G>&);
  TEST_FUNC friend void iter_swap(const PointerTo<G>&, const PointerTo<F>&);
};

static_assert(cuda::std::indirectly_swappable<PointerTo<A>, PointerTo<B>>);
static_assert(!cuda::std::indirectly_swappable<PointerTo<A>, PointerTo<C>>);
static_assert(!cuda::std::indirectly_swappable<PointerTo<A>, PointerTo<D>>);
static_assert(!cuda::std::indirectly_swappable<PointerTo<A>, PointerTo<E>>);
static_assert(!cuda::std::indirectly_swappable<PointerTo<A>, PointerTo<G>>);

int main(int, char**)
{
  return 0;
}
