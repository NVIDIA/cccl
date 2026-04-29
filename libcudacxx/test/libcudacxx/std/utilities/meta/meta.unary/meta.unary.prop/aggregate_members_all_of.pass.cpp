//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__type_traits/aggregate_members_all_of.h>
#include <cuda/std/type_traits>

#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// test types

struct Empty
{};

struct OneMember
{
  int x;
};

struct TwoMembers
{
  int x;
  float y;
};

struct ThreeMembers
{
  int x;
  float y;
  double z;
};

struct AllInts
{
  int a;
  int b;
  int c;
};

struct LargeAggregate
{
  int m0;
  int m1;
  int m2;
  int m3;
  int m4;
  int m5;
  int m6;
  int m7;
};

struct TooLarge
{
  int m0;
  int m1;
  int m2;
  int m3;
  int m4;
  int m5;
  int m6;
  int m7;
  int m8;
};

struct Nested
{
  OneMember inner;
  int extra;
};

struct WithArray
{
  int values[4];
};

class NonAggregate
{
public:
  __host__ __device__ NonAggregate() {}
  int x;
};

struct NonTriviallyCopyable
{
  int x;

  NonTriviallyCopyable() = default;

  __host__ __device__ NonTriviallyCopyable(const NonTriviallyCopyable&) {}
};

struct HasNonTriviallyCopyableMember
{
  int x;
  NonTriviallyCopyable y;
};

// Edge case: aggregate with a deleted default constructor.
// Without the special handling for N=0, compilers (nvc++, nvcc+clang) do not agree on the arity
struct NoDefaultCtor
{
  NoDefaultCtor() = delete;
  __host__ __device__ NoDefaultCtor(int) {}
  int value;
};

struct AggregateWithDeletedDefault
{
  NoDefaultCtor m;
};

//----------------------------------------------------------------------------------------------------------------------
// predicates

template <typename T>
struct always_true : cuda::std::true_type
{};

template <typename T>
struct always_false : cuda::std::false_type
{};

//----------------------------------------------------------------------------------------------------------------------
// __aggregate_arity_v tests

static_assert(cuda::std::__aggregate_arity_v<Empty> == 0);
static_assert(cuda::std::__aggregate_arity_v<OneMember> == 1);
static_assert(cuda::std::__aggregate_arity_v<TwoMembers> == 2);
static_assert(cuda::std::__aggregate_arity_v<ThreeMembers> == 3);
static_assert(cuda::std::__aggregate_arity_v<AllInts> == 3);
static_assert(cuda::std::__aggregate_arity_v<LargeAggregate> == 8);

static_assert(cuda::std::is_aggregate_v<AggregateWithDeletedDefault>);
static_assert(cuda::std::__aggregate_arity_v<AggregateWithDeletedDefault> == 1);
#if !TEST_COMPILER(MSVC) // MSVC does not perform brace elision in SFINAE contexts
static_assert(cuda::std::__aggregate_arity_v<Nested> == 2);
static_assert(cuda::std::__aggregate_arity_v<WithArray> == 4);
#endif // !TEST_COMPILER(MSVC)

//----------------------------------------------------------------------------------------------------------------------
// __aggregate_all_of_v tests

// empty aggregate: always true
static_assert(cuda::std::__aggregate_all_of_v<always_true, Empty>);
static_assert(cuda::std::__aggregate_all_of_v<always_false, Empty>);

// all members satisfy the predicate
static_assert(cuda::std::__aggregate_all_of_v<cuda::std::is_integral, AllInts>);
static_assert(cuda::std::__aggregate_all_of_v<always_true, ThreeMembers>);

// not all members satisfy the predicate
static_assert(!cuda::std::__aggregate_all_of_v<cuda::std::is_integral, TwoMembers>); // mixed int/float
static_assert(!cuda::std::__aggregate_all_of_v<always_false, OneMember>);

// max arity aggregate
static_assert(cuda::std::__aggregate_all_of_v<cuda::std::is_integral, LargeAggregate>);

// too many members: returns false (arity exceeds __aggregate_max_arity)
static_assert(!cuda::std::__aggregate_all_of_v<cuda::std::is_integral, TooLarge>);

// nested / array members
#if !TEST_COMPILER(MSVC) // MSVC does not perform brace elision in SFINAE contexts
static_assert(cuda::std::__aggregate_all_of_v<cuda::std::is_integral, Nested>);
static_assert(cuda::std::__aggregate_all_of_v<cuda::std::is_integral, WithArray>);
static_assert(cuda::std::__aggregate_all_of_v<cuda::std::is_trivially_copyable, Nested>);
static_assert(cuda::std::__aggregate_all_of_v<cuda::std::is_trivially_copyable, WithArray>);
#endif // !TEST_COMPILER(MSVC)

// non-aggregate: always false
static_assert(!cuda::std::__aggregate_all_of_v<always_true, NonAggregate>);
static_assert(!cuda::std::__aggregate_all_of_v<cuda::std::is_integral, NonAggregate>);

// aggregate with a non-trivially-copyable member
static_assert(!cuda::std::__aggregate_all_of_v<cuda::std::is_trivially_copyable, HasNonTriviallyCopyableMember>);

// Edge case: aggregate with a deleted default constructor.
// Without the special handling for N=0, compilers do not agree on the arity
// With the special case: arity == 1
static_assert(cuda::std::__aggregate_all_of_v<always_true, AggregateWithDeletedDefault>);
static_assert(!cuda::std::__aggregate_all_of_v<always_false, AggregateWithDeletedDefault>);
#if !TEST_COMPILER(MSVC) || TEST_STD_VER > 2017
static_assert(!cuda::std::__aggregate_all_of_v<cuda::std::is_integral, AggregateWithDeletedDefault>);
#endif // !TEST_COMPILER(MSVC) || TEST_STD_VER > 2017

int main(int, char**)
{
  return 0;
}
