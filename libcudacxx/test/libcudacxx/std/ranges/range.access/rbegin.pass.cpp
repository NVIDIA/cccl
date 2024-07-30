//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// cuda::std::ranges::rbegin
// cuda::std::ranges::crbegin

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"

using RangeRBeginT  = decltype(cuda::std::ranges::rbegin);
using RangeCRBeginT = decltype(cuda::std::ranges::crbegin);

STATIC_TEST_GLOBAL_VAR int globalBuff[8];

static_assert(!cuda::std::is_invocable_v<RangeRBeginT, int (&&)[10]>);
static_assert(cuda::std::is_invocable_v<RangeRBeginT, int (&)[10]>);
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, int (&&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, int (&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, int (&&)[10]>);
static_assert(cuda::std::is_invocable_v<RangeCRBeginT, int (&)[10]>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, int (&&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, int (&)[]>);

struct Incomplete;

static_assert(!cuda::std::is_invocable_v<RangeRBeginT, Incomplete (&&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, const Incomplete (&&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, Incomplete (&&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, const Incomplete (&&)[]>);

static_assert(!cuda::std::is_invocable_v<RangeRBeginT, Incomplete (&&)[10]>);
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, const Incomplete (&&)[10]>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, Incomplete (&&)[10]>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, const Incomplete (&&)[10]>);

// This case is IFNDR; we handle it SFINAE-friendly.
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, Incomplete (&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, const Incomplete (&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, Incomplete (&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, const Incomplete (&)[]>);

// This case is IFNDR; we handle it SFINAE-friendly.
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, Incomplete (&)[10]>);
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, const Incomplete (&)[10]>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, Incomplete (&)[10]>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, const Incomplete (&)[10]>);

struct RBeginMember
{
  int x;
  __host__ __device__ constexpr const int* rbegin() const
  {
    return &x;
  }
};

// Ensure that we can't call with rvalues with borrowing disabled.
static_assert(cuda::std::is_invocable_v<RangeRBeginT, RBeginMember&>);
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, RBeginMember&&>);
static_assert(cuda::std::is_invocable_v<RangeRBeginT, RBeginMember const&>);
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, RBeginMember const&&>);
static_assert(cuda::std::is_invocable_v<RangeCRBeginT, RBeginMember&>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, RBeginMember&&>);
static_assert(cuda::std::is_invocable_v<RangeCRBeginT, RBeginMember const&>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, RBeginMember const&&>);

struct Different
{
  __host__ __device__ char*& rbegin();
  __host__ __device__ short*& rbegin() const;
};

__host__ __device__ constexpr bool testReturnTypes()
{
  {
    int* x[2] = {};
    unused(x);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::rbegin(x)), cuda::std::reverse_iterator<int**>);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::crbegin(x)), cuda::std::reverse_iterator<int* const*>);
  }
  {
    int x[2][2] = {};
    unused(x);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::rbegin(x)), cuda::std::reverse_iterator<int(*)[2]>);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::crbegin(x)), cuda::std::reverse_iterator<const int(*)[2]>);
  }
  {
    Different x{};
    unused(x);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::rbegin(x)), char*);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::crbegin(x)), short*);
  }
  return true;
}

__host__ __device__ constexpr bool testArray()
{
  int a[2] = {};
  assert(cuda::std::ranges::rbegin(a).base() == a + 2);
  assert(cuda::std::ranges::crbegin(a).base() == a + 2);

  int b[2][2] = {};
  assert(cuda::std::ranges::rbegin(b).base() == b + 2);
  assert(cuda::std::ranges::crbegin(b).base() == b + 2);

  RBeginMember c[2] = {};
  assert(cuda::std::ranges::rbegin(c).base() == c + 2);
  assert(cuda::std::ranges::crbegin(c).base() == c + 2);

  return true;
}

struct RBeginMemberReturnsInt
{
  __host__ __device__ int rbegin() const;
};
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, RBeginMemberReturnsInt const&>);

struct RBeginMemberReturnsVoidPtr
{
  __host__ __device__ const void* rbegin() const;
};
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, RBeginMemberReturnsVoidPtr const&>);

struct PtrConvertibleRBeginMember
{
  struct iterator
  {
    __host__ __device__ operator int*() const;
  };
  __host__ __device__ iterator rbegin() const;
};
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, PtrConvertibleRBeginMember const&>);

struct NonConstRBeginMember
{
  int x;
  __host__ __device__ constexpr int* rbegin()
  {
    return &x;
  }
};
static_assert(cuda::std::is_invocable_v<RangeRBeginT, NonConstRBeginMember&>);
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, NonConstRBeginMember const&>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, NonConstRBeginMember&>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, NonConstRBeginMember const&>);

struct EnabledBorrowingRBeginMember
{
  __host__ __device__ constexpr int* rbegin() const
  {
    return globalBuff;
  }
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<EnabledBorrowingRBeginMember> = true;

struct RBeginMemberFunction
{
  int x;
  __host__ __device__ constexpr const int* rbegin() const
  {
    return &x;
  }
  __host__ __device__ friend int* rbegin(RBeginMemberFunction const&);
};

struct EmptyPtrRBeginMember
{
  struct Empty
  {};
  Empty x;
  __host__ __device__ constexpr const Empty* rbegin() const
  {
    return &x;
  }
};

__host__ __device__ constexpr bool testRBeginMember()
{
  RBeginMember a{};
  assert(cuda::std::ranges::rbegin(a) == &a.x);
  assert(cuda::std::ranges::crbegin(a) == &a.x);
  static_assert(!cuda::std::is_invocable_v<RangeRBeginT, RBeginMember&&>);
  static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, RBeginMember&&>);

  NonConstRBeginMember b{};
  assert(cuda::std::ranges::rbegin(b) == &b.x);
  static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, NonConstRBeginMember&>);

  EnabledBorrowingRBeginMember c{};
  assert(cuda::std::ranges::rbegin(c) == globalBuff);
  assert(cuda::std::ranges::crbegin(c) == globalBuff);
  assert(cuda::std::ranges::rbegin(cuda::std::move(c)) == globalBuff);
  assert(cuda::std::ranges::crbegin(cuda::std::move(c)) == globalBuff);

  RBeginMemberFunction d{};
  assert(cuda::std::ranges::rbegin(d) == &d.x);
  assert(cuda::std::ranges::crbegin(d) == &d.x);

  EmptyPtrRBeginMember e{};
  assert(cuda::std::ranges::rbegin(e) == &e.x);
  assert(cuda::std::ranges::crbegin(e) == &e.x);

  return true;
}

struct RBeginFunction
{
  int x;
  __host__ __device__ friend constexpr const int* rbegin(RBeginFunction const& bf)
  {
    return &bf.x;
  }
};
static_assert(cuda::std::is_invocable_v<RangeRBeginT, RBeginFunction const&>);
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, RBeginFunction&&>);
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, RBeginFunction&>);
static_assert(cuda::std::is_invocable_v<RangeCRBeginT, RBeginFunction const&>);
static_assert(cuda::std::is_invocable_v<RangeCRBeginT, RBeginFunction&>);

struct RBeginFunctionReturnsInt
{
  __host__ __device__ friend int rbegin(RBeginFunctionReturnsInt const&);
};
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, RBeginFunctionReturnsInt const&>);

struct RBeginFunctionReturnsVoidPtr
{
  __host__ __device__ friend void* rbegin(RBeginFunctionReturnsVoidPtr const&);
};
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, RBeginFunctionReturnsVoidPtr const&>);

struct RBeginFunctionReturnsEmpty
{
  struct Empty
  {};
  __host__ __device__ friend Empty rbegin(RBeginFunctionReturnsEmpty const&);
};
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, RBeginFunctionReturnsEmpty const&>);

struct RBeginFunctionReturnsPtrConvertible
{
  struct iterator
  {
    __host__ __device__ operator int*() const;
  };
  __host__ __device__ friend iterator rbegin(RBeginFunctionReturnsPtrConvertible const&);
};
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, RBeginFunctionReturnsPtrConvertible const&>);

struct RBeginFunctionByValue
{
  __host__ __device__ friend constexpr int* rbegin(RBeginFunctionByValue)
  {
    return globalBuff + 1;
  }
};
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, RBeginFunctionByValue>);

struct RBeginFunctionEnabledBorrowing
{
  __host__ __device__ friend constexpr int* rbegin(RBeginFunctionEnabledBorrowing)
  {
    return globalBuff + 2;
  }
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<RBeginFunctionEnabledBorrowing> = true;

struct RBeginFunctionReturnsEmptyPtr
{
  struct Empty
  {};
  Empty x;
  __host__ __device__ friend constexpr const Empty* rbegin(RBeginFunctionReturnsEmptyPtr const& bf)
  {
    return &bf.x;
  }
};

struct RBeginFunctionWithDataMember
{
  int x;
  int rbegin;
  __host__ __device__ friend constexpr const int* rbegin(RBeginFunctionWithDataMember const& bf)
  {
    return &bf.x;
  }
};

struct RBeginFunctionWithPrivateBeginMember
{
  int y;
  __host__ __device__ friend constexpr const int* rbegin(RBeginFunctionWithPrivateBeginMember const& bf)
  {
    return &bf.y;
  }

private:
  __host__ __device__ const int* rbegin() const;
};

__host__ __device__ constexpr bool testRBeginFunction()
{
  RBeginFunction a{};
  const RBeginFunction aa{};
  static_assert(!cuda::std::invocable<RangeRBeginT, decltype((a))>);
  assert(cuda::std::ranges::crbegin(a) == &a.x);
  assert(cuda::std::ranges::rbegin(aa) == &aa.x);
  assert(cuda::std::ranges::crbegin(aa) == &aa.x);

  RBeginFunctionByValue b{};
  const RBeginFunctionByValue bb{};
  assert(cuda::std::ranges::rbegin(b) == globalBuff + 1);
  assert(cuda::std::ranges::crbegin(b) == globalBuff + 1);
  assert(cuda::std::ranges::rbegin(bb) == globalBuff + 1);
  assert(cuda::std::ranges::crbegin(bb) == globalBuff + 1);

  RBeginFunctionEnabledBorrowing c{};
  const RBeginFunctionEnabledBorrowing cc{};
  assert(cuda::std::ranges::rbegin(cuda::std::move(c)) == globalBuff + 2);
  assert(cuda::std::ranges::crbegin(cuda::std::move(c)) == globalBuff + 2);
  assert(cuda::std::ranges::rbegin(cuda::std::move(cc)) == globalBuff + 2);
  assert(cuda::std::ranges::crbegin(cuda::std::move(cc)) == globalBuff + 2);

  RBeginFunctionReturnsEmptyPtr d{};
  const RBeginFunctionReturnsEmptyPtr dd{};
  static_assert(!cuda::std::invocable<RangeRBeginT, decltype((d))>);
  assert(cuda::std::ranges::crbegin(d) == &d.x);
  assert(cuda::std::ranges::rbegin(dd) == &dd.x);
  assert(cuda::std::ranges::crbegin(dd) == &dd.x);

  RBeginFunctionWithDataMember e{};
  const RBeginFunctionWithDataMember ee{};
  static_assert(!cuda::std::invocable<RangeRBeginT, decltype((e))>);
  assert(cuda::std::ranges::rbegin(ee) == &ee.x);
  assert(cuda::std::ranges::crbegin(e) == &e.x);
  assert(cuda::std::ranges::crbegin(ee) == &ee.x);

  RBeginFunctionWithPrivateBeginMember f{};
  const RBeginFunctionWithPrivateBeginMember ff{};
  static_assert(!cuda::std::invocable<RangeRBeginT, decltype((f))>);
  assert(cuda::std::ranges::crbegin(f) == &f.y);
  assert(cuda::std::ranges::rbegin(ff) == &ff.y);
  assert(cuda::std::ranges::crbegin(ff) == &ff.y);

  return true;
}

struct MemberBeginEnd
{
  int b, e;
  char cb, ce;
  __host__ __device__ constexpr bidirectional_iterator<int*> begin()
  {
    return bidirectional_iterator<int*>(&b);
  }
  __host__ __device__ constexpr bidirectional_iterator<int*> end()
  {
    return bidirectional_iterator<int*>(&e);
  }
  __host__ __device__ constexpr bidirectional_iterator<const char*> begin() const
  {
    return bidirectional_iterator<const char*>(&cb);
  }
  __host__ __device__ constexpr bidirectional_iterator<const char*> end() const
  {
    return bidirectional_iterator<const char*>(&ce);
  }
};
static_assert(cuda::std::is_invocable_v<RangeRBeginT, MemberBeginEnd&>);
static_assert(cuda::std::is_invocable_v<RangeRBeginT, MemberBeginEnd const&>);
static_assert(cuda::std::is_invocable_v<RangeCRBeginT, MemberBeginEnd const&>);

struct FunctionBeginEnd
{
  int b, e;
  char cb, ce;
  __host__ __device__ friend constexpr bidirectional_iterator<int*> begin(FunctionBeginEnd& v)
  {
    return bidirectional_iterator<int*>(&v.b);
  }
  __host__ __device__ friend constexpr bidirectional_iterator<int*> end(FunctionBeginEnd& v)
  {
    return bidirectional_iterator<int*>(&v.e);
  }
  __host__ __device__ friend constexpr bidirectional_iterator<const char*> begin(const FunctionBeginEnd& v)
  {
    return bidirectional_iterator<const char*>(&v.cb);
  }
  __host__ __device__ friend constexpr bidirectional_iterator<const char*> end(const FunctionBeginEnd& v)
  {
    return bidirectional_iterator<const char*>(&v.ce);
  }
};
static_assert(cuda::std::is_invocable_v<RangeRBeginT, FunctionBeginEnd&>);
static_assert(cuda::std::is_invocable_v<RangeRBeginT, FunctionBeginEnd const&>);
static_assert(cuda::std::is_invocable_v<RangeCRBeginT, FunctionBeginEnd const&>);

struct MemberBeginFunctionEnd
{
  int b, e;
  char cb, ce;
  __host__ __device__ constexpr bidirectional_iterator<int*> begin()
  {
    return bidirectional_iterator<int*>(&b);
  }
  __host__ __device__ friend constexpr bidirectional_iterator<int*> end(MemberBeginFunctionEnd& v)
  {
    return bidirectional_iterator<int*>(&v.e);
  }
  __host__ __device__ constexpr bidirectional_iterator<const char*> begin() const
  {
    return bidirectional_iterator<const char*>(&cb);
  }
  __host__ __device__ friend constexpr bidirectional_iterator<const char*> end(const MemberBeginFunctionEnd& v)
  {
    return bidirectional_iterator<const char*>(&v.ce);
  }
};
static_assert(cuda::std::is_invocable_v<RangeRBeginT, MemberBeginFunctionEnd&>);
static_assert(cuda::std::is_invocable_v<RangeRBeginT, MemberBeginFunctionEnd const&>);
static_assert(cuda::std::is_invocable_v<RangeCRBeginT, MemberBeginFunctionEnd const&>);

struct FunctionBeginMemberEnd
{
  int b, e;
  char cb, ce;
  __host__ __device__ friend constexpr bidirectional_iterator<int*> begin(FunctionBeginMemberEnd& v)
  {
    return bidirectional_iterator<int*>(&v.b);
  }
  __host__ __device__ constexpr bidirectional_iterator<int*> end()
  {
    return bidirectional_iterator<int*>(&e);
  }
  __host__ __device__ friend constexpr bidirectional_iterator<const char*> begin(const FunctionBeginMemberEnd& v)
  {
    return bidirectional_iterator<const char*>(&v.cb);
  }
  __host__ __device__ constexpr bidirectional_iterator<const char*> end() const
  {
    return bidirectional_iterator<const char*>(&ce);
  }
};
static_assert(cuda::std::is_invocable_v<RangeRBeginT, FunctionBeginMemberEnd&>);
static_assert(cuda::std::is_invocable_v<RangeRBeginT, FunctionBeginMemberEnd const&>);
static_assert(cuda::std::is_invocable_v<RangeCRBeginT, FunctionBeginMemberEnd const&>);

struct MemberBeginEndDifferentTypes
{
  __host__ __device__ bidirectional_iterator<int*> begin();
  __host__ __device__ bidirectional_iterator<const int*> end();
};
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, MemberBeginEndDifferentTypes&>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, MemberBeginEndDifferentTypes&>);

struct FunctionBeginEndDifferentTypes
{
  __host__ __device__ friend bidirectional_iterator<int*> begin(FunctionBeginEndDifferentTypes&);
  __host__ __device__ friend bidirectional_iterator<const int*> end(FunctionBeginEndDifferentTypes&);
};
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, FunctionBeginEndDifferentTypes&>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, FunctionBeginEndDifferentTypes&>);

struct MemberBeginEndForwardIterators
{
  __host__ __device__ forward_iterator<int*> begin();
  __host__ __device__ forward_iterator<int*> end();
};
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, MemberBeginEndForwardIterators&>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, MemberBeginEndForwardIterators&>);

struct FunctionBeginEndForwardIterators
{
  __host__ __device__ friend forward_iterator<int*> begin(FunctionBeginEndForwardIterators&);
  __host__ __device__ friend forward_iterator<int*> end(FunctionBeginEndForwardIterators&);
};
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, FunctionBeginEndForwardIterators&>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, FunctionBeginEndForwardIterators&>);

struct MemberBeginOnly
{
  __host__ __device__ bidirectional_iterator<int*> begin() const;
};
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, MemberBeginOnly&>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, MemberBeginOnly&>);

struct FunctionBeginOnly
{
  __host__ __device__ friend bidirectional_iterator<int*> begin(FunctionBeginOnly&);
};
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, FunctionBeginOnly&>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, FunctionBeginOnly&>);

struct MemberEndOnly
{
  __host__ __device__ bidirectional_iterator<int*> end() const;
};
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, MemberEndOnly&>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, MemberEndOnly&>);

struct FunctionEndOnly
{
  __host__ __device__ friend bidirectional_iterator<int*> end(FunctionEndOnly&);
};
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, FunctionEndOnly&>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, FunctionEndOnly&>);

// Make sure there is no clash between the following cases:
// - the case that handles classes defining member `rbegin` and `rend` functions;
// - the case that handles classes defining `begin` and `end` functions returning reversible iterators.
struct MemberBeginAndRBegin
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
  __host__ __device__ int* rbegin() const;
  __host__ __device__ int* rend() const;
};
static_assert(cuda::std::is_invocable_v<RangeRBeginT, MemberBeginAndRBegin&>);
static_assert(cuda::std::is_invocable_v<RangeCRBeginT, MemberBeginAndRBegin&>);
static_assert(cuda::std::same_as<cuda::std::invoke_result_t<RangeRBeginT, MemberBeginAndRBegin&>, int*>);
static_assert(cuda::std::same_as<cuda::std::invoke_result_t<RangeCRBeginT, MemberBeginAndRBegin&>, int*>);

__host__ __device__ constexpr bool testBeginEnd()
{
  MemberBeginEnd a{};
  const MemberBeginEnd aa{};
  assert(base(cuda::std::ranges::rbegin(a).base()) == &a.e);
  assert(base(cuda::std::ranges::crbegin(a).base()) == &a.ce);
  assert(base(cuda::std::ranges::rbegin(aa).base()) == &aa.ce);
  assert(base(cuda::std::ranges::crbegin(aa).base()) == &aa.ce);

  FunctionBeginEnd b{};
  const FunctionBeginEnd bb{};
  assert(base(cuda::std::ranges::rbegin(b).base()) == &b.e);
  assert(base(cuda::std::ranges::crbegin(b).base()) == &b.ce);
  assert(base(cuda::std::ranges::rbegin(bb).base()) == &bb.ce);
  assert(base(cuda::std::ranges::crbegin(bb).base()) == &bb.ce);

  MemberBeginFunctionEnd c{};
  const MemberBeginFunctionEnd cc{};
  assert(base(cuda::std::ranges::rbegin(c).base()) == &c.e);
  assert(base(cuda::std::ranges::crbegin(c).base()) == &c.ce);
  assert(base(cuda::std::ranges::rbegin(cc).base()) == &cc.ce);
  assert(base(cuda::std::ranges::crbegin(cc).base()) == &cc.ce);

  FunctionBeginMemberEnd d{};
  const FunctionBeginMemberEnd dd{};
  assert(base(cuda::std::ranges::rbegin(d).base()) == &d.e);
  assert(base(cuda::std::ranges::crbegin(d).base()) == &d.ce);
  assert(base(cuda::std::ranges::rbegin(dd).base()) == &dd.ce);
  assert(base(cuda::std::ranges::crbegin(dd).base()) == &dd.ce);

  return true;
}
ASSERT_NOEXCEPT(cuda::std::ranges::rbegin(cuda::std::declval<int (&)[10]>()));
ASSERT_NOEXCEPT(cuda::std::ranges::crbegin(cuda::std::declval<int (&)[10]>()));

#if !defined(TEST_COMPILER_MSVC_2019) // broken noexcept
_CCCL_GLOBAL_CONSTANT struct NoThrowMemberRBegin
{
  __host__ __device__ ThrowingIterator<int> rbegin() const noexcept; // auto(t.rbegin()) doesn't throw
} ntmb;
static_assert(noexcept(cuda::std::ranges::rbegin(ntmb)));
static_assert(noexcept(cuda::std::ranges::crbegin(ntmb)));

_CCCL_GLOBAL_CONSTANT struct NoThrowADLRBegin
{
  __host__ __device__ friend ThrowingIterator<int> rbegin(NoThrowADLRBegin&) noexcept; // auto(rbegin(t)) doesn't throw
  __host__ __device__ friend ThrowingIterator<int> rbegin(const NoThrowADLRBegin&) noexcept;
} ntab;
static_assert(noexcept(cuda::std::ranges::rbegin(ntab)));
static_assert(noexcept(cuda::std::ranges::crbegin(ntab)));
#endif // !TEST_COMPILER_MSVC_2019

#if !defined(TEST_COMPILER_ICC)
_CCCL_GLOBAL_CONSTANT struct NoThrowMemberRBeginReturnsRef
{
  __host__ __device__ ThrowingIterator<int>& rbegin() const noexcept; // auto(t.rbegin()) may throw
} ntmbrr;
static_assert(!noexcept(cuda::std::ranges::rbegin(ntmbrr)));
static_assert(!noexcept(cuda::std::ranges::crbegin(ntmbrr)));
#endif // !TEST_COMPILER_ICC

_CCCL_GLOBAL_CONSTANT struct RBeginReturnsArrayRef
{
  __host__ __device__ auto rbegin() const noexcept -> int (&)[10];
} brar;
static_assert(noexcept(cuda::std::ranges::rbegin(brar)));
static_assert(noexcept(cuda::std::ranges::crbegin(brar)));

#if !defined(TEST_COMPILER_ICC)
_CCCL_GLOBAL_CONSTANT struct NoThrowBeginThrowingEnd
{
  __host__ __device__ int* begin() const noexcept;
  __host__ __device__ int* end() const;
} ntbte;
static_assert(!noexcept(cuda::std::ranges::rbegin(ntbte)));
static_assert(!noexcept(cuda::std::ranges::crbegin(ntbte)));
#endif // !TEST_COMPILER_ICC

_CCCL_GLOBAL_CONSTANT struct NoThrowEndThrowingBegin
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const noexcept;
} ntetb;
static_assert(noexcept(cuda::std::ranges::rbegin(ntetb)));
static_assert(noexcept(cuda::std::ranges::crbegin(ntetb)));

#if TEST_STD_VER > 2017
// Test ADL-proofing.
struct Incomplete;
template <class T>
struct Holder
{
  T t;
};
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, Holder<Incomplete>*>);
static_assert(!cuda::std::is_invocable_v<RangeRBeginT, Holder<Incomplete>*&>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, Holder<Incomplete>*>);
static_assert(!cuda::std::is_invocable_v<RangeCRBeginT, Holder<Incomplete>*&>);
#endif // TEST_STD_VER > 2017

int main(int, char**)
{
  static_assert(testReturnTypes());

  testArray();
#ifndef TEST_COMPILER_CUDACC_BELOW_11_3
  static_assert(testArray());
#endif // TEST_COMPILER_CUDACC_BELOW_11_3

  testRBeginMember();
  static_assert(testRBeginMember());

  testRBeginFunction();
  static_assert(testRBeginFunction());

  testBeginEnd();
  static_assert(testBeginEnd());

#if !defined(TEST_COMPILER_MSVC_2019)
  unused(ntmb);
  unused(ntab);
#endif // !TEST_COMPILER_MSVC_2019
#if !defined(TEST_COMPILER_ICC)
  unused(ntmbrr);
#endif // !TEST_COMPILER_ICC
  unused(brar);
#if !defined(TEST_COMPILER_ICC)
  unused(ntbte);
#endif // !TEST_COMPILER_ICC
  unused(ntetb);

  return 0;
}
