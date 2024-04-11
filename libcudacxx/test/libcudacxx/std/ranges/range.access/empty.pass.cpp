//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// cuda::std::ranges::empty

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"

using RangeEmptyT = decltype(cuda::std::ranges::empty);

static_assert(!cuda::std::is_invocable_v<RangeEmptyT, int[]>);
static_assert(!cuda::std::is_invocable_v<RangeEmptyT, int (&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeEmptyT, int (&&)[]>);
static_assert(cuda::std::is_invocable_v<RangeEmptyT, int[1]>);
static_assert(cuda::std::is_invocable_v<RangeEmptyT, const int[1]>);
static_assert(cuda::std::is_invocable_v<RangeEmptyT, int (&&)[1]>);
static_assert(cuda::std::is_invocable_v<RangeEmptyT, int (&)[1]>);
static_assert(cuda::std::is_invocable_v<RangeEmptyT, const int (&)[1]>);

struct Incomplete;
static_assert(!cuda::std::is_invocable_v<RangeEmptyT, Incomplete[]>);
static_assert(!cuda::std::is_invocable_v<RangeEmptyT, Incomplete (&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeEmptyT, Incomplete (&&)[]>);

#ifndef TEST_COMPILER_NVRTC
extern Incomplete array_of_incomplete[42];
static_assert(!cuda::std::ranges::empty(array_of_incomplete));
static_assert(!cuda::std::ranges::empty(cuda::std::move(array_of_incomplete)));

extern const Incomplete const_array_of_incomplete[42];
static_assert(!cuda::std::ranges::empty(const_array_of_incomplete));
static_assert(!cuda::std::ranges::empty(static_cast<const Incomplete (&&)[42]>(array_of_incomplete)));
#endif // TEST_COMPILER_NVRTC

struct InputRangeWithoutSize
{
  __host__ __device__ cpp17_input_iterator<int*> begin() const;
  __host__ __device__ cpp17_input_iterator<int*> end() const;
};
static_assert(!cuda::std::is_invocable_v<RangeEmptyT, const InputRangeWithoutSize&>);

struct NonConstEmpty
{
  __host__ __device__ bool empty();
};
static_assert(!cuda::std::is_invocable_v<RangeEmptyT, const NonConstEmpty&>);

struct HasMemberAndFunction
{
  __host__ __device__ constexpr bool empty() const
  {
    return true;
  }
  // We should never do ADL lookup for cuda::std::ranges::empty.
  __host__ __device__ friend bool empty(const HasMemberAndFunction&)
  {
    return false;
  }
};

struct BadReturnType
{
  __host__ __device__ BadReturnType empty()
  {
    return {};
  }
};
static_assert(!cuda::std::is_invocable_v<RangeEmptyT, BadReturnType&>);

struct BoolConvertible
{
  __host__ __device__ constexpr explicit operator bool() noexcept(false)
  {
    return true;
  }
};
struct BoolConvertibleReturnType
{
  __host__ __device__ constexpr BoolConvertible empty() noexcept
  {
    return {};
  }
};
// old GCC seems to fall over the chaining of the noexcept clauses here
#if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 9) && !defined(TEST_COMPILER_MSVC) && !defined(TEST_COMPILER_ICC)
static_assert(!noexcept(cuda::std::ranges::empty(BoolConvertibleReturnType())));
#endif // (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 9)

struct InputIterators
{
  __host__ __device__ cpp17_input_iterator<int*> begin() const;
  __host__ __device__ cpp17_input_iterator<int*> end() const;
};
static_assert(cuda::std::is_same_v<decltype(InputIterators().begin() == InputIterators().end()), bool>);
static_assert(!cuda::std::is_invocable_v<RangeEmptyT, const InputIterators&>);

__host__ __device__ constexpr bool testEmptyMember()
{
  HasMemberAndFunction a{};
  assert(cuda::std::ranges::empty(a));

  BoolConvertibleReturnType b{};
  assert(cuda::std::ranges::empty(b));

  return true;
}

struct SizeMember
{
  size_t size_;
  __host__ __device__ constexpr size_t size() const
  {
    return size_;
  }
};

struct SizeFunction
{
  size_t size_;
  __host__ __device__ friend constexpr size_t size(SizeFunction sf)
  {
    return sf.size_;
  }
};

struct BeginEndSizedSentinel
{
  __host__ __device__ constexpr int* begin() const
  {
    return nullptr;
  }
  __host__ __device__ constexpr auto end() const
  {
    return sized_sentinel<int*>(nullptr);
  }
};
static_assert(cuda::std::ranges::forward_range<BeginEndSizedSentinel>);
static_assert(cuda::std::ranges::sized_range<BeginEndSizedSentinel>);

__host__ __device__ constexpr bool testUsingRangesSize()
{
  SizeMember a{1};
  assert(!cuda::std::ranges::empty(a));
  SizeMember b{0};
  assert(cuda::std::ranges::empty(b));

  SizeFunction c{1};
  assert(!cuda::std::ranges::empty(c));
  SizeFunction d{0};
  assert(cuda::std::ranges::empty(d));

  BeginEndSizedSentinel e{};
  assert(cuda::std::ranges::empty(e));

  return true;
}

struct BeginEndNotSizedSentinel
{
  __host__ __device__ constexpr int* begin() const
  {
    return nullptr;
  }
  __host__ __device__ constexpr auto end() const
  {
    return sentinel_wrapper<int*>(nullptr);
  }
};
static_assert(cuda::std::ranges::forward_range<BeginEndNotSizedSentinel>);
static_assert(!cuda::std::ranges::sized_range<BeginEndNotSizedSentinel>);

// size is disabled here, so we have to compare begin and end.
struct DisabledSizeRangeWithBeginEnd
{
  __host__ __device__ constexpr int* begin() const
  {
    return nullptr;
  }
  __host__ __device__ constexpr auto end() const
  {
    return sentinel_wrapper<int*>(nullptr);
  }
  __host__ __device__ size_t size() const;
};
template <>
inline constexpr bool cuda::std::ranges::disable_sized_range<DisabledSizeRangeWithBeginEnd> = true;
static_assert(cuda::std::ranges::contiguous_range<DisabledSizeRangeWithBeginEnd>);
static_assert(!cuda::std::ranges::sized_range<DisabledSizeRangeWithBeginEnd>);

struct BeginEndAndEmpty
{
  __host__ __device__ constexpr int* begin() const
  {
    return nullptr;
  }
  __host__ __device__ constexpr auto end() const
  {
    return sentinel_wrapper<int*>(nullptr);
  }
  __host__ __device__ constexpr bool empty()
  {
    return false;
  }
};

struct EvilBeginEnd
{
  __host__ __device__ bool empty() &&;
  __host__ __device__ constexpr int* begin() &
  {
    return nullptr;
  }
  __host__ __device__ constexpr int* end() &
  {
    return nullptr;
  }
};

__host__ __device__ constexpr bool testBeginEqualsEnd()
{
  BeginEndNotSizedSentinel a{};
  assert(cuda::std::ranges::empty(a));

  DisabledSizeRangeWithBeginEnd d{};
  assert(cuda::std::ranges::empty(d));

  BeginEndAndEmpty e{};
  assert(!cuda::std::ranges::empty(e)); // e.empty()

  const BeginEndAndEmpty ce{};
  assert(cuda::std::ranges::empty(ce)); // e.begin() == e.end()

  assert(cuda::std::ranges::empty(EvilBeginEnd()));

  return true;
}

#if TEST_STD_VER > 2017
// Test ADL-proofing.
struct Incomplete;
template <class T>
struct Holder
{
  T t;
};
static_assert(!cuda::std::is_invocable_v<RangeEmptyT, Holder<Incomplete>*>);
static_assert(!cuda::std::is_invocable_v<RangeEmptyT, Holder<Incomplete>*&>);
#endif

int main(int, char**)
{
  testEmptyMember();
  static_assert(testEmptyMember());

  testUsingRangesSize();
  static_assert(testUsingRangesSize());

  testBeginEqualsEnd();
  static_assert(testBeginEqualsEnd());

  return 0;
}
