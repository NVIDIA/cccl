//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class F, class I>
// concept indirectly_regular_unary_invocable;

#include <cuda/std/concepts>
#include <cuda/std/iterator>

#include "indirectly_readable.h"

using It = IndirectlyReadable<struct Token>;
using R1 = T1<struct ReturnToken>;
using R2 = T2<struct ReturnToken>;

template <class I>
struct GoodInvocable
{
  __host__ __device__ R1 operator()(cuda::std::iter_value_t<I>&) const;
  __host__ __device__ R2 operator()(cuda::std::iter_reference_t<I>) const;
  __host__ __device__ R2 operator()(cuda::std::iter_common_reference_t<I>) const;
};

// Should work when all constraints are satisfied
static_assert(cuda::std::indirectly_regular_unary_invocable<GoodInvocable<It>, It>);

// Should fail when the iterator is not indirectly_readable
#if TEST_STD_VER > 2017
struct NotIndirectlyReadable
{};
static_assert(
  !cuda::std::indirectly_regular_unary_invocable<GoodInvocable<NotIndirectlyReadable>, NotIndirectlyReadable>);
#endif

// Should fail when the invocable is not copy constructible
struct BadInvocable1
{
  BadInvocable1(BadInvocable1 const&) = delete;
  template <class T>
  __host__ __device__ R1 operator()(T const&) const;
};
static_assert(!cuda::std::indirectly_regular_unary_invocable<BadInvocable1, It>);

// Should fail when the invocable can't be called with (iter_value_t&)
struct BadInvocable2
{
  template <class T>
  __host__ __device__ R1 operator()(T const&) const;
  R1 operator()(cuda::std::iter_value_t<It>&) const = delete;
};
static_assert(!cuda::std::indirectly_regular_unary_invocable<BadInvocable2, It>);

// Should fail when the invocable can't be called with (iter_reference_t)
struct BadInvocable3
{
  template <class T>
  __host__ __device__ R1 operator()(T const&) const;
  R1 operator()(cuda::std::iter_reference_t<It>) const = delete;
};
static_assert(!cuda::std::indirectly_regular_unary_invocable<BadInvocable3, It>);

// Should fail when the invocable can't be called with (iter_common_reference_t)
struct BadInvocable4
{
  template <class T>
  __host__ __device__ R1 operator()(T const&) const;
  R1 operator()(cuda::std::iter_common_reference_t<It>) const = delete;
};
static_assert(!cuda::std::indirectly_regular_unary_invocable<BadInvocable4, It>);

// Should fail when the invocable doesn't have a common reference between its return types
struct BadInvocable5
{
  __host__ __device__ R1 operator()(cuda::std::iter_value_t<It>&) const;
  struct Unrelated
  {};
  __host__ __device__ Unrelated operator()(cuda::std::iter_reference_t<It>) const;
  __host__ __device__ R1 operator()(cuda::std::iter_common_reference_t<It>) const;
};
static_assert(!cuda::std::indirectly_regular_unary_invocable<BadInvocable5, It>);

// Various tests with callables
struct S
{};
static_assert(cuda::std::indirectly_regular_unary_invocable<int (*)(int), int*>);
static_assert(cuda::std::indirectly_regular_unary_invocable<int (&)(int), int*>);
static_assert(cuda::std::indirectly_regular_unary_invocable<int S::*, S*>);
static_assert(cuda::std::indirectly_regular_unary_invocable<int (S::*)(), S*>);
static_assert(cuda::std::indirectly_regular_unary_invocable<int (S::*)() const, S*>);
static_assert(cuda::std::indirectly_regular_unary_invocable<void (*)(int), int*>);

static_assert(!cuda::std::indirectly_regular_unary_invocable<int(int), int*>); // not move constructible
static_assert(!cuda::std::indirectly_regular_unary_invocable<int (*)(int*, int*), int*>);
static_assert(!cuda::std::indirectly_regular_unary_invocable<int (&)(int*, int*), int*>);
static_assert(!cuda::std::indirectly_regular_unary_invocable<int (S::*)(int*), S*>);
static_assert(!cuda::std::indirectly_regular_unary_invocable<int (S::*)(int*) const, S*>);

int main(int, char**)
{
  return 0;
}
