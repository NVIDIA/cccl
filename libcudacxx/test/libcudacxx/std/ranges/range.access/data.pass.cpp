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

// cuda::std::ranges::data

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

using RangeDataT  = decltype(cuda::std::ranges::data);
using RangeCDataT = decltype(cuda::std::ranges::cdata);

STATIC_TEST_GLOBAL_VAR int globalBuff[2] = {};

struct Incomplete;

static_assert(!cuda::std::is_invocable_v<RangeDataT, Incomplete[]>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, Incomplete (&&)[2]>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, Incomplete (&&)[2][2]>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, int[1]>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, int (&&)[1]>);
static_assert(cuda::std::is_invocable_v<RangeDataT, int (&)[1]>);

static_assert(!cuda::std::is_invocable_v<RangeCDataT, Incomplete[]>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, Incomplete (&&)[2]>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, Incomplete (&&)[2][2]>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, int[1]>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, int (&&)[1]>);
static_assert(cuda::std::is_invocable_v<RangeCDataT, int (&)[1]>);

struct DataMember
{
  int x;
  __host__ __device__ constexpr const int* data() const
  {
    return &x;
  }
};
static_assert(cuda::std::is_invocable_v<RangeDataT, DataMember&>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, DataMember&&>);
static_assert(cuda::std::is_invocable_v<RangeDataT, DataMember const&>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, DataMember const&&>);
static_assert(cuda::std::is_invocable_v<RangeCDataT, DataMember&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, DataMember&&>);
static_assert(cuda::std::is_invocable_v<RangeCDataT, DataMember const&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, DataMember const&&>);

struct D
{
  __host__ __device__ char*& data();
  __host__ __device__ short*& data() const;
};

struct NC
{
  __host__ __device__ char* begin() const;
  __host__ __device__ char* end() const;
  __host__ __device__ int* data();
};

__host__ __device__ constexpr bool testReturnTypes()
{
  {
    int* x[2] = {};
    unused(x);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::data(x)), int**);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::cdata(x)), int* const*);
  }
  {
    int x[2][2] = {};
    unused(x);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::data(x)), int(*)[2]);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::cdata(x)), const int(*)[2]);
  }
  {
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::data(cuda::std::declval<D&>())), char*);
    static_assert(!cuda::std::is_invocable_v<RangeDataT, D&&>);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::data(cuda::std::declval<const D&>())), short*);
    static_assert(!cuda::std::is_invocable_v<RangeDataT, const D&&>);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::cdata(cuda::std::declval<D&>())), short*);
    static_assert(!cuda::std::is_invocable_v<RangeCDataT, D&&>);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::cdata(cuda::std::declval<const D&>())), short*);
    static_assert(!cuda::std::is_invocable_v<RangeCDataT, const D&&>);
  }
  {
    static_assert(!cuda::std::ranges::contiguous_range<NC>);
    static_assert(cuda::std::ranges::contiguous_range<const NC>);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::data(cuda::std::declval<NC&>())), int*);
    static_assert(!cuda::std::is_invocable_v<RangeDataT, NC&&>);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::data(cuda::std::declval<const NC&>())), char*);
    static_assert(!cuda::std::is_invocable_v<RangeDataT, const NC&&>);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::cdata(cuda::std::declval<NC&>())), char*);
    static_assert(!cuda::std::is_invocable_v<RangeCDataT, NC&&>);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::cdata(cuda::std::declval<const NC&>())), char*);
    static_assert(!cuda::std::is_invocable_v<RangeCDataT, const NC&&>);
  }
  return true;
}

struct VoidDataMember
{
  __host__ __device__ void* data() const;
};
static_assert(!cuda::std::is_invocable_v<RangeDataT, VoidDataMember const&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, VoidDataMember const&>);

struct Empty
{};
struct EmptyDataMember
{
  __host__ __device__ Empty data() const;
};
static_assert(!cuda::std::is_invocable_v<RangeDataT, EmptyDataMember const&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, EmptyDataMember const&>);

struct PtrConvertibleDataMember
{
  struct Ptr
  {
    __host__ __device__ operator int*() const;
  };
  __host__ __device__ Ptr data() const;
};
static_assert(!cuda::std::is_invocable_v<RangeDataT, PtrConvertibleDataMember const&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, PtrConvertibleDataMember const&>);

struct NonConstDataMember
{
  int x;
  __host__ __device__ constexpr int* data()
  {
    return &x;
  }
};

struct EnabledBorrowingDataMember
{
  __host__ __device__ constexpr const int* data()
  {
    return &globalBuff[0];
  }
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<EnabledBorrowingDataMember> = true;

struct DataMemberAndBegin
{
  int x;
  __host__ __device__ constexpr const int* data() const
  {
    return &x;
  }
  __host__ __device__ const int* begin() const;
};

__host__ __device__ constexpr bool testDataMember()
{
  DataMember a{};
  assert(cuda::std::ranges::data(a) == &a.x);
  assert(cuda::std::ranges::cdata(a) == &a.x);

  NonConstDataMember b{};
  assert(cuda::std::ranges::data(b) == &b.x);
  static_assert(!cuda::std::is_invocable_v<RangeCDataT, decltype((b))>);

  EnabledBorrowingDataMember c{};
  assert(cuda::std::ranges::data(cuda::std::move(c)) == &globalBuff[0]);
  static_assert(!cuda::std::is_invocable_v<RangeCDataT, decltype(cuda::std::move(c))>);

  DataMemberAndBegin d{};
  assert(cuda::std::ranges::data(d) == &d.x);
  assert(cuda::std::ranges::cdata(d) == &d.x);

  return true;
}

using ContiguousIter = contiguous_iterator<const int*>;

struct BeginMemberContiguousIterator
{
  int buff[8];

  __host__ __device__ constexpr ContiguousIter begin() const
  {
    return ContiguousIter(buff);
  }
};
static_assert(cuda::std::is_invocable_v<RangeDataT, BeginMemberContiguousIterator&>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, BeginMemberContiguousIterator&&>);
static_assert(cuda::std::is_invocable_v<RangeDataT, BeginMemberContiguousIterator const&>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, BeginMemberContiguousIterator const&&>);
static_assert(cuda::std::is_invocable_v<RangeCDataT, BeginMemberContiguousIterator&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, BeginMemberContiguousIterator&&>);
static_assert(cuda::std::is_invocable_v<RangeCDataT, BeginMemberContiguousIterator const&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, BeginMemberContiguousIterator const&&>);

struct BeginMemberRandomAccess
{
  int buff[8];

  __host__ __device__ random_access_iterator<const int*> begin() const;
};
static_assert(!cuda::std::is_invocable_v<RangeDataT, BeginMemberRandomAccess&>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, BeginMemberRandomAccess&&>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, const BeginMemberRandomAccess&>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, const BeginMemberRandomAccess&&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, BeginMemberRandomAccess&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, BeginMemberRandomAccess&&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, const BeginMemberRandomAccess&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, const BeginMemberRandomAccess&&>);

struct BeginFriendContiguousIterator
{
  int buff[8];

  __host__ __device__ friend constexpr ContiguousIter begin(const BeginFriendContiguousIterator& iter)
  {
    return ContiguousIter(iter.buff);
  }
};
static_assert(cuda::std::is_invocable_v<RangeDataT, BeginMemberContiguousIterator&>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, BeginMemberContiguousIterator&&>);
static_assert(cuda::std::is_invocable_v<RangeDataT, BeginMemberContiguousIterator const&>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, BeginMemberContiguousIterator const&&>);
static_assert(cuda::std::is_invocable_v<RangeCDataT, BeginMemberContiguousIterator&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, BeginMemberContiguousIterator&&>);
static_assert(cuda::std::is_invocable_v<RangeCDataT, BeginMemberContiguousIterator const&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, BeginMemberContiguousIterator const&&>);

struct BeginFriendRandomAccess
{
  __host__ __device__ friend random_access_iterator<const int*> begin(const BeginFriendRandomAccess iter);
};
static_assert(!cuda::std::is_invocable_v<RangeDataT, BeginFriendRandomAccess&>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, BeginFriendRandomAccess&&>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, const BeginFriendRandomAccess&>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, const BeginFriendRandomAccess&&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, BeginFriendRandomAccess&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, BeginFriendRandomAccess&&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, const BeginFriendRandomAccess&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, const BeginFriendRandomAccess&&>);

struct BeginMemberRvalue
{
  int buff[8];

  __host__ __device__ ContiguousIter begin() &&;
};
static_assert(!cuda::std::is_invocable_v<RangeDataT, BeginMemberRvalue&>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, BeginMemberRvalue&&>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, BeginMemberRvalue const&>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, BeginMemberRvalue const&&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, BeginMemberRvalue&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, BeginMemberRvalue&&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, BeginMemberRvalue const&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, BeginMemberRvalue const&&>);

struct BeginMemberBorrowingEnabled
{
  __host__ __device__ constexpr contiguous_iterator<const int*> begin()
  {
    return contiguous_iterator<const int*>{&globalBuff[1]};
  }
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<BeginMemberBorrowingEnabled> = true;

static_assert(cuda::std::is_invocable_v<RangeDataT, BeginMemberBorrowingEnabled&>);
static_assert(cuda::std::is_invocable_v<RangeDataT, BeginMemberBorrowingEnabled&&>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, BeginMemberBorrowingEnabled const&>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, BeginMemberBorrowingEnabled const&&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, BeginMemberBorrowingEnabled&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, BeginMemberBorrowingEnabled&&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, BeginMemberBorrowingEnabled const&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, BeginMemberBorrowingEnabled const&&>);

__host__ __device__ constexpr bool testViaRangesBegin()
{
  int arr[2] = {};
  assert(cuda::std::ranges::data(arr) == arr + 0);
  assert(cuda::std::ranges::cdata(arr) == arr + 0);

  BeginMemberContiguousIterator a{};
  assert(cuda::std::ranges::data(a) == a.buff);
  assert(cuda::std::ranges::cdata(a) == a.buff);

  const BeginFriendContiguousIterator b{};
  assert(cuda::std::ranges::data(b) == b.buff);
  assert(cuda::std::ranges::cdata(b) == b.buff);

  BeginMemberBorrowingEnabled c{};
  assert(cuda::std::ranges::data(cuda::std::move(c)) == &globalBuff[1]);
  static_assert(!cuda::std::is_invocable_v<RangeCDataT, decltype(cuda::std::move(c))>);

  return true;
}

#if TEST_STD_VER >= 2020
// Test ADL-proofing.
struct Incomplete;
template <class T>
struct Holder
{
  T t;
};
static_assert(!cuda::std::is_invocable_v<RangeDataT, Holder<Incomplete>*>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, Holder<Incomplete>*&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, Holder<Incomplete>*>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, Holder<Incomplete>*&>);
#endif // TEST_STD_VER >= 2020

struct RandomButNotContiguous
{
  __host__ __device__ random_access_iterator<int*> begin() const;
  __host__ __device__ random_access_iterator<int*> end() const;
};
static_assert(!cuda::std::is_invocable_v<RangeDataT, RandomButNotContiguous>);
static_assert(!cuda::std::is_invocable_v<RangeDataT, RandomButNotContiguous&>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, RandomButNotContiguous>);
static_assert(!cuda::std::is_invocable_v<RangeCDataT, RandomButNotContiguous&>);

int main(int, char**)
{
  static_assert(testReturnTypes());

  testDataMember();
  static_assert(testDataMember());

  testViaRangesBegin();
  static_assert(testViaRangesBegin());

  return 0;
}
