//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class I1, class I2>
// concept indirectly_swappable;

#include <cuda/std/iterator>

#include "test_macros.h"

template <class T, class ValueType = T>
struct PointerTo
{
  using value_type = ValueType;
  __host__ __device__ T& operator*() const;
};

static_assert(cuda::std::indirectly_swappable<PointerTo<int>>);
static_assert(cuda::std::indirectly_swappable<PointerTo<int>, PointerTo<int>>);

struct B;

struct A
{
  __host__ __device__ friend void iter_swap(const PointerTo<A>&, const PointerTo<A>&);
};

// Is indirectly swappable.
struct B
{
  __host__ __device__ friend void iter_swap(const PointerTo<B>&, const PointerTo<B>&);
  __host__ __device__ friend void iter_swap(const PointerTo<A>&, const PointerTo<B>&);
  __host__ __device__ friend void iter_swap(const PointerTo<B>&, const PointerTo<A>&);
};

// Valid except ranges::iter_swap(i2, i1).
struct C
{
  __host__ __device__ friend void iter_swap(const PointerTo<C>&, const PointerTo<C>&);
  __host__ __device__ friend void iter_swap(const PointerTo<A>&, const PointerTo<C>&);
  __host__ __device__ friend void iter_swap(const PointerTo<C>&, const PointerTo<A>&) = delete;
};

// Valid except ranges::iter_swap(i1, i2).
struct D
{
  __host__ __device__ friend void iter_swap(const PointerTo<D>&, const PointerTo<D>&);
  __host__ __device__ friend void iter_swap(const PointerTo<A>&, const PointerTo<D>&) = delete;
  __host__ __device__ friend void iter_swap(const PointerTo<D>&, const PointerTo<A>&);
};

// Valid except ranges::iter_swap(i2, i2).
struct E
{
  E operator=(const E&)                                                               = delete;
  __host__ __device__ friend void iter_swap(const PointerTo<E>&, const PointerTo<E>&) = delete;
  __host__ __device__ friend void iter_swap(const PointerTo<A>&, const PointerTo<E>&);
  __host__ __device__ friend void iter_swap(const PointerTo<E>&, const PointerTo<A>&);
};

struct F
{
  __host__ __device__ friend void iter_swap(const PointerTo<F>&, const PointerTo<F>&) = delete;
};

// Valid except ranges::iter_swap(i1, i1).
struct G
{
  __host__ __device__ friend void iter_swap(const PointerTo<G>&, const PointerTo<G>&);
  __host__ __device__ friend void iter_swap(const PointerTo<F>&, const PointerTo<G>&);
  __host__ __device__ friend void iter_swap(const PointerTo<G>&, const PointerTo<F>&);
};

#if !defined(TEST_COMPILER_CUDACC_BELOW_11_3) && !defined(TEST_COMPILER_MSVC_2017)
static_assert(cuda::std::indirectly_swappable<PointerTo<A>, PointerTo<B>>);
#endif // !TEST_COMPILER_CUDACC_BELOW_11_3 && !TEST_COMPILER_MSVC_2017
static_assert(!cuda::std::indirectly_swappable<PointerTo<A>, PointerTo<C>>);
static_assert(!cuda::std::indirectly_swappable<PointerTo<A>, PointerTo<D>>);
static_assert(!cuda::std::indirectly_swappable<PointerTo<A>, PointerTo<E>>);
static_assert(!cuda::std::indirectly_swappable<PointerTo<A>, PointerTo<G>>);

int main(int, char**)
{
  return 0;
}
