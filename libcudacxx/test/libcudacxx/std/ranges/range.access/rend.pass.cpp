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

// cuda::std::ranges::rend
// cuda::std::ranges::crend

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"

using RangeREndT  = decltype(cuda::std::ranges::rend);
using RangeCREndT = decltype(cuda::std::ranges::crend);

STATIC_TEST_GLOBAL_VAR int globalBuff[8];

static_assert(!cuda::std::is_invocable_v<RangeREndT, int (&&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeREndT, int (&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeREndT, int (&&)[10]>);
static_assert(cuda::std::is_invocable_v<RangeREndT, int (&)[10]>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, int (&&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, int (&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, int (&&)[10]>);
static_assert(cuda::std::is_invocable_v<RangeCREndT, int (&)[10]>);

struct Incomplete;
static_assert(!cuda::std::is_invocable_v<RangeREndT, Incomplete (&&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeREndT, Incomplete (&&)[42]>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, Incomplete (&&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, Incomplete (&&)[42]>);

struct REndMember
{
  int x;
  __host__ __device__ const int* rbegin() const;
  __host__ __device__ constexpr const int* rend() const
  {
    return &x;
  }
};

// Ensure that we can't call with rvalues with borrowing disabled.
static_assert(cuda::std::is_invocable_v<RangeREndT, REndMember&>);
static_assert(!cuda::std::is_invocable_v<RangeREndT, REndMember&&>);
static_assert(cuda::std::is_invocable_v<RangeREndT, REndMember const&>);
static_assert(!cuda::std::is_invocable_v<RangeREndT, REndMember const&&>);
static_assert(cuda::std::is_invocable_v<RangeCREndT, REndMember&>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, REndMember&&>);
static_assert(cuda::std::is_invocable_v<RangeCREndT, REndMember const&>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, REndMember const&&>);

struct Different
{
  __host__ __device__ char* rbegin();
  __host__ __device__ sentinel_wrapper<char*>& rend();
  __host__ __device__ short* rbegin() const;
  __host__ __device__ sentinel_wrapper<short*>& rend() const;
};

__host__ __device__ constexpr bool testReturnTypes()
{
  {
    int* x[2] = {};
    unused(x);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::rend(x)), cuda::std::reverse_iterator<int**>);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::crend(x)), cuda::std::reverse_iterator<int* const*>);
  }

  {
    int x[2][2] = {};
    unused(x);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::rend(x)), cuda::std::reverse_iterator<int(*)[2]>);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::crend(x)), cuda::std::reverse_iterator<const int(*)[2]>);
  }

  {
    Different x{};
    unused(x);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::rend(x)), sentinel_wrapper<char*>);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::crend(x)), sentinel_wrapper<short*>);
  }

  return true;
}

__host__ __device__ TEST_CONSTEXPR_CXX17 bool testArray()
{
  int a[2] = {};
  assert(cuda::std::ranges::rend(a).base() == a);
  assert(cuda::std::ranges::crend(a).base() == a);

  int b[2][2] = {};
  assert(cuda::std::ranges::rend(b).base() == b);
  assert(cuda::std::ranges::crend(b).base() == b);

  REndMember c[2] = {};
  assert(cuda::std::ranges::rend(c).base() == c);
  assert(cuda::std::ranges::crend(c).base() == c);

  return true;
}

struct REndMemberReturnsInt
{
  __host__ __device__ int rbegin() const;
  __host__ __device__ int rend() const;
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, REndMemberReturnsInt const&>);

struct REndMemberReturnsVoidPtr
{
  __host__ __device__ const void* rbegin() const;
  __host__ __device__ const void* rend() const;
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, REndMemberReturnsVoidPtr const&>);

struct PtrConvertible
{
  __host__ __device__ operator int*() const;
};
struct PtrConvertibleREndMember
{
  __host__ __device__ PtrConvertible rbegin() const;
  __host__ __device__ PtrConvertible rend() const;
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, PtrConvertibleREndMember const&>);

struct NoRBeginMember
{
  __host__ __device__ constexpr const int* rend();
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, NoRBeginMember const&>);

struct NonConstREndMember
{
  int x;
  __host__ __device__ constexpr int* rbegin()
  {
    return nullptr;
  }
  __host__ __device__ constexpr int* rend()
  {
    return &x;
  }
};
static_assert(cuda::std::is_invocable_v<RangeREndT, NonConstREndMember&>);
static_assert(!cuda::std::is_invocable_v<RangeREndT, NonConstREndMember const&>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, NonConstREndMember&>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, NonConstREndMember const&>);

struct EnabledBorrowingREndMember
{
  __host__ __device__ constexpr int* rbegin() const
  {
    return nullptr;
  }
  __host__ __device__ constexpr int* rend() const
  {
    return &globalBuff[0];
  }
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<EnabledBorrowingREndMember> = true;

struct REndMemberFunction
{
  int x;
  __host__ __device__ constexpr const int* rbegin() const
  {
    return nullptr;
  }
  __host__ __device__ constexpr const int* rend() const
  {
    return &x;
  }
  __host__ __device__ friend constexpr int* rend(REndMemberFunction const&);
};

struct Empty
{};
struct EmptyEndMember
{
  __host__ __device__ Empty rbegin() const;
  __host__ __device__ Empty rend() const;
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, EmptyEndMember const&>);

struct EmptyPtrREndMember
{
  Empty x;
  __host__ __device__ constexpr const Empty* rbegin() const
  {
    return nullptr;
  }
  __host__ __device__ constexpr const Empty* rend() const
  {
    return &x;
  }
};

__host__ __device__ constexpr bool testREndMember()
{
  REndMember a{};
  assert(cuda::std::ranges::rend(a) == &a.x);
  assert(cuda::std::ranges::crend(a) == &a.x);

  NonConstREndMember b{};
  assert(cuda::std::ranges::rend(b) == &b.x);
  static_assert(!cuda::std::is_invocable_v<RangeCREndT, decltype((b))>);

  EnabledBorrowingREndMember c{};
  assert(cuda::std::ranges::rend(cuda::std::move(c)) == &globalBuff[0]);
  assert(cuda::std::ranges::crend(cuda::std::move(c)) == &globalBuff[0]);

  REndMemberFunction d{};
  assert(cuda::std::ranges::rend(d) == &d.x);
  assert(cuda::std::ranges::crend(d) == &d.x);

  EmptyPtrREndMember e{};
  assert(cuda::std::ranges::rend(e) == &e.x);
  assert(cuda::std::ranges::crend(e) == &e.x);

  return true;
}

struct REndFunction
{
  int x;
  __host__ __device__ friend constexpr const int* rbegin(REndFunction const&)
  {
    return nullptr;
  }
  __host__ __device__ friend constexpr const int* rend(REndFunction const& bf)
  {
    return &bf.x;
  }
};

static_assert(cuda::std::is_invocable_v<RangeREndT, REndFunction const&>);
static_assert(!cuda::std::is_invocable_v<RangeREndT, REndFunction&&>);

static_assert(cuda::std::is_invocable_v<RangeREndT, REndFunction const&>);
static_assert(!cuda::std::is_invocable_v<RangeREndT, REndFunction&&>);
static_assert(!cuda::std::is_invocable_v<RangeREndT, REndFunction&>);
static_assert(cuda::std::is_invocable_v<RangeCREndT, REndFunction const&>);
static_assert(cuda::std::is_invocable_v<RangeCREndT, REndFunction&>);

struct REndFunctionReturnsInt
{
  __host__ __device__ friend constexpr int rbegin(REndFunctionReturnsInt const&);
  __host__ __device__ friend constexpr int rend(REndFunctionReturnsInt const&);
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, REndFunctionReturnsInt const&>);

struct REndFunctionReturnsVoidPtr
{
  __host__ __device__ friend constexpr void* rbegin(REndFunctionReturnsVoidPtr const&);
  __host__ __device__ friend constexpr void* rend(REndFunctionReturnsVoidPtr const&);
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, REndFunctionReturnsVoidPtr const&>);

struct REndFunctionReturnsEmpty
{
  __host__ __device__ friend constexpr Empty rbegin(REndFunctionReturnsEmpty const&);
  __host__ __device__ friend constexpr Empty rend(REndFunctionReturnsEmpty const&);
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, REndFunctionReturnsEmpty const&>);

struct REndFunctionReturnsPtrConvertible
{
  __host__ __device__ friend constexpr PtrConvertible rbegin(REndFunctionReturnsPtrConvertible const&);
  __host__ __device__ friend constexpr PtrConvertible rend(REndFunctionReturnsPtrConvertible const&);
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, REndFunctionReturnsPtrConvertible const&>);

struct NoRBeginFunction
{
  __host__ __device__ friend constexpr const int* rend(NoRBeginFunction const&);
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, NoRBeginFunction const&>);

struct REndFunctionByValue
{
  __host__ __device__ friend constexpr int* rbegin(REndFunctionByValue)
  {
    return nullptr;
  }
  __host__ __device__ friend constexpr int* rend(REndFunctionByValue)
  {
    return &globalBuff[1];
  }
};
static_assert(!cuda::std::is_invocable_v<RangeCREndT, REndFunctionByValue>);

struct REndFunctionEnabledBorrowing
{
  __host__ __device__ friend constexpr int* rbegin(REndFunctionEnabledBorrowing)
  {
    return nullptr;
  }
  __host__ __device__ friend constexpr int* rend(REndFunctionEnabledBorrowing)
  {
    return &globalBuff[2];
  }
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<REndFunctionEnabledBorrowing> = true;

struct REndFunctionReturnsEmptyPtr
{
  Empty x;
  __host__ __device__ friend constexpr const Empty* rbegin(REndFunctionReturnsEmptyPtr const&)
  {
    return nullptr;
  }
  __host__ __device__ friend constexpr const Empty* rend(REndFunctionReturnsEmptyPtr const& bf)
  {
    return &bf.x;
  }
};

struct REndFunctionWithDataMember
{
  int x;
  int rend;
  __host__ __device__ friend constexpr const int* rbegin(REndFunctionWithDataMember const&)
  {
    return nullptr;
  }
  __host__ __device__ friend constexpr const int* rend(REndFunctionWithDataMember const& bf)
  {
    return &bf.x;
  }
};

struct REndFunctionWithPrivateEndMember : private REndMember
{
  int y;
  __host__ __device__ friend constexpr const int* rbegin(REndFunctionWithPrivateEndMember const&)
  {
    return nullptr;
  }
  __host__ __device__ friend constexpr const int* rend(REndFunctionWithPrivateEndMember const& bf)
  {
    return &bf.y;
  }
};

struct RBeginMemberEndFunction
{
  int x;
  __host__ __device__ constexpr const int* rbegin() const
  {
    return nullptr;
  }
  __host__ __device__ friend constexpr const int* rend(RBeginMemberEndFunction const& bf)
  {
    return &bf.x;
  }
};

__host__ __device__ constexpr bool testREndFunction()
{
  const REndFunction a{};
  assert(cuda::std::ranges::rend(a) == &a.x);
  assert(cuda::std::ranges::crend(a) == &a.x);
  REndFunction aa{};
  static_assert(!cuda::std::is_invocable_v<RangeREndT, decltype((aa))>);
  assert(cuda::std::ranges::crend(aa) == &aa.x);

  REndFunctionByValue b{};
  assert(cuda::std::ranges::rend(b) == &globalBuff[1]);
  assert(cuda::std::ranges::crend(b) == &globalBuff[1]);

  REndFunctionEnabledBorrowing c{};
  assert(cuda::std::ranges::rend(cuda::std::move(c)) == &globalBuff[2]);
  assert(cuda::std::ranges::crend(cuda::std::move(c)) == &globalBuff[2]);

  const REndFunctionReturnsEmptyPtr d{};
  assert(cuda::std::ranges::rend(d) == &d.x);
  assert(cuda::std::ranges::crend(d) == &d.x);
  REndFunctionReturnsEmptyPtr dd{};
  static_assert(!cuda::std::is_invocable_v<RangeREndT, decltype((dd))>);
  assert(cuda::std::ranges::crend(dd) == &dd.x);

  const REndFunctionWithDataMember e{};
  assert(cuda::std::ranges::rend(e) == &e.x);
  assert(cuda::std::ranges::crend(e) == &e.x);
  REndFunctionWithDataMember ee{};
  static_assert(!cuda::std::is_invocable_v<RangeREndT, decltype((ee))>);
  assert(cuda::std::ranges::crend(ee) == &ee.x);

  const REndFunctionWithPrivateEndMember f{};
  assert(cuda::std::ranges::rend(f) == &f.y);
  assert(cuda::std::ranges::crend(f) == &f.y);
  REndFunctionWithPrivateEndMember ff{};
  static_assert(!cuda::std::is_invocable_v<RangeREndT, decltype((ff))>);
  assert(cuda::std::ranges::crend(ff) == &ff.y);

  const RBeginMemberEndFunction g{};
  assert(cuda::std::ranges::rend(g) == &g.x);
  assert(cuda::std::ranges::crend(g) == &g.x);
  RBeginMemberEndFunction gg{};
  static_assert(!cuda::std::is_invocable_v<RangeREndT, decltype((gg))>);
  assert(cuda::std::ranges::crend(gg) == &gg.x);

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
static_assert(cuda::std::is_invocable_v<RangeREndT, MemberBeginEnd&>);
static_assert(cuda::std::is_invocable_v<RangeREndT, MemberBeginEnd const&>);
static_assert(cuda::std::is_invocable_v<RangeCREndT, MemberBeginEnd const&>);

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
static_assert(cuda::std::is_invocable_v<RangeREndT, FunctionBeginEnd&>);
static_assert(cuda::std::is_invocable_v<RangeREndT, FunctionBeginEnd const&>);
static_assert(cuda::std::is_invocable_v<RangeCREndT, FunctionBeginEnd const&>);

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
static_assert(cuda::std::is_invocable_v<RangeREndT, MemberBeginFunctionEnd&>);
static_assert(cuda::std::is_invocable_v<RangeREndT, MemberBeginFunctionEnd const&>);
static_assert(cuda::std::is_invocable_v<RangeCREndT, MemberBeginFunctionEnd const&>);

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
static_assert(cuda::std::is_invocable_v<RangeREndT, FunctionBeginMemberEnd&>);
static_assert(cuda::std::is_invocable_v<RangeREndT, FunctionBeginMemberEnd const&>);
static_assert(cuda::std::is_invocable_v<RangeCREndT, FunctionBeginMemberEnd const&>);

struct MemberBeginEndDifferentTypes
{
  __host__ __device__ bidirectional_iterator<int*> begin();
  __host__ __device__ bidirectional_iterator<const int*> end();
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, MemberBeginEndDifferentTypes&>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, MemberBeginEndDifferentTypes&>);

struct FunctionBeginEndDifferentTypes
{
  __host__ __device__ friend bidirectional_iterator<int*> begin(FunctionBeginEndDifferentTypes&);
  __host__ __device__ friend bidirectional_iterator<const int*> end(FunctionBeginEndDifferentTypes&);
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, FunctionBeginEndDifferentTypes&>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, FunctionBeginEndDifferentTypes&>);

struct MemberBeginEndForwardIterators
{
  __host__ __device__ forward_iterator<int*> begin();
  __host__ __device__ forward_iterator<int*> end();
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, MemberBeginEndForwardIterators&>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, MemberBeginEndForwardIterators&>);

struct FunctionBeginEndForwardIterators
{
  __host__ __device__ friend forward_iterator<int*> begin(FunctionBeginEndForwardIterators&);
  __host__ __device__ friend forward_iterator<int*> end(FunctionBeginEndForwardIterators&);
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, FunctionBeginEndForwardIterators&>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, FunctionBeginEndForwardIterators&>);

struct MemberBeginOnly
{
  __host__ __device__ bidirectional_iterator<int*> begin() const;
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, MemberBeginOnly&>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, MemberBeginOnly&>);

struct FunctionBeginOnly
{
  __host__ __device__ friend bidirectional_iterator<int*> begin(FunctionBeginOnly&);
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, FunctionBeginOnly&>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, FunctionBeginOnly&>);

struct MemberEndOnly
{
  __host__ __device__ bidirectional_iterator<int*> end() const;
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, MemberEndOnly&>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, MemberEndOnly&>);

struct FunctionEndOnly
{
  __host__ __device__ friend bidirectional_iterator<int*> end(FunctionEndOnly&);
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, FunctionEndOnly&>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, FunctionEndOnly&>);

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
static_assert(cuda::std::is_invocable_v<RangeREndT, MemberBeginAndRBegin&>);
static_assert(cuda::std::is_invocable_v<RangeCREndT, MemberBeginAndRBegin&>);
static_assert(cuda::std::same_as<cuda::std::invoke_result_t<RangeREndT, MemberBeginAndRBegin&>, int*>);
static_assert(cuda::std::same_as<cuda::std::invoke_result_t<RangeCREndT, MemberBeginAndRBegin&>, int*>);

__host__ __device__ TEST_CONSTEXPR_CXX17 bool testBeginEnd()
{
  MemberBeginEnd a{};
  const MemberBeginEnd aa{};
  assert(base(cuda::std::ranges::rend(a).base()) == &a.b);
  assert(base(cuda::std::ranges::crend(a).base()) == &a.cb);
  assert(base(cuda::std::ranges::rend(aa).base()) == &aa.cb);
  assert(base(cuda::std::ranges::crend(aa).base()) == &aa.cb);

  FunctionBeginEnd b{};
  const FunctionBeginEnd bb{};
  assert(base(cuda::std::ranges::rend(b).base()) == &b.b);
  assert(base(cuda::std::ranges::crend(b).base()) == &b.cb);
  assert(base(cuda::std::ranges::rend(bb).base()) == &bb.cb);
  assert(base(cuda::std::ranges::crend(bb).base()) == &bb.cb);

  MemberBeginFunctionEnd c{};
  const MemberBeginFunctionEnd cc{};
  assert(base(cuda::std::ranges::rend(c).base()) == &c.b);
  assert(base(cuda::std::ranges::crend(c).base()) == &c.cb);
  assert(base(cuda::std::ranges::rend(cc).base()) == &cc.cb);
  assert(base(cuda::std::ranges::crend(cc).base()) == &cc.cb);

  FunctionBeginMemberEnd d{};
  const FunctionBeginMemberEnd dd{};
  assert(base(cuda::std::ranges::rend(d).base()) == &d.b);
  assert(base(cuda::std::ranges::crend(d).base()) == &d.cb);
  assert(base(cuda::std::ranges::rend(dd).base()) == &dd.cb);
  assert(base(cuda::std::ranges::crend(dd).base()) == &dd.cb);

  return true;
}
ASSERT_NOEXCEPT(cuda::std::ranges::rend(cuda::std::declval<int (&)[10]>()));
ASSERT_NOEXCEPT(cuda::std::ranges::crend(cuda::std::declval<int (&)[10]>()));

#if !defined(TEST_COMPILER_MSVC_2019)
_CCCL_GLOBAL_CONSTANT struct NoThrowMemberREnd
{
  __host__ __device__ ThrowingIterator<int> rbegin() const;
  __host__ __device__ ThrowingIterator<int> rend() const noexcept; // auto(t.rend()) doesn't throw
} ntmre;
static_assert(noexcept(cuda::std::ranges::rend(ntmre)));
static_assert(noexcept(cuda::std::ranges::crend(ntmre)));

_CCCL_GLOBAL_CONSTANT struct NoThrowADLREnd
{
  __host__ __device__ ThrowingIterator<int> rbegin() const;
  __host__ __device__ friend ThrowingIterator<int> rend(NoThrowADLREnd&) noexcept; // auto(rend(t)) doesn't throw
  __host__ __device__ friend ThrowingIterator<int> rend(const NoThrowADLREnd&) noexcept;
} ntare;
static_assert(noexcept(cuda::std::ranges::rend(ntare)));
static_assert(noexcept(cuda::std::ranges::crend(ntare)));
#endif // !TEST_COMPILER_MSVC_2019

#if !defined(TEST_COMPILER_ICC)
_CCCL_GLOBAL_CONSTANT struct NoThrowMemberREndReturnsRef
{
  __host__ __device__ ThrowingIterator<int> rbegin() const;
  __host__ __device__ ThrowingIterator<int>& rend() const noexcept; // auto(t.rend()) may throw
} ntmrerr;
static_assert(!noexcept(cuda::std::ranges::rend(ntmrerr)));
static_assert(!noexcept(cuda::std::ranges::crend(ntmrerr)));
#endif // !TEST_COMPILER_ICC

_CCCL_GLOBAL_CONSTANT struct REndReturnsArrayRef
{
  __host__ __device__ auto rbegin() const noexcept -> int (&)[10];
  __host__ __device__ auto rend() const noexcept -> int (&)[10];
} rerar;
static_assert(noexcept(cuda::std::ranges::rend(rerar)));
static_assert(noexcept(cuda::std::ranges::crend(rerar)));

_CCCL_GLOBAL_CONSTANT struct NoThrowBeginThrowingEnd
{
  __host__ __device__ int* begin() const noexcept;
  __host__ __device__ int* end() const;
} ntbte;
static_assert(noexcept(cuda::std::ranges::rend(ntbte)));
static_assert(noexcept(cuda::std::ranges::crend(ntbte)));

#if !defined(TEST_COMPILER_ICC)
_CCCL_GLOBAL_CONSTANT struct NoThrowEndThrowingBegin
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const noexcept;
} ntetb;
static_assert(!noexcept(cuda::std::ranges::rend(ntetb)));
static_assert(!noexcept(cuda::std::ranges::crend(ntetb)));
#endif // !TEST_COMPILER_ICC

#if TEST_STD_VER > 2017
// Test ADL-proofing.
struct Incomplete;
template <class T>
struct Holder
{
  T t;
};
static_assert(!cuda::std::is_invocable_v<RangeREndT, Holder<Incomplete>*>);
static_assert(!cuda::std::is_invocable_v<RangeREndT, Holder<Incomplete>*&>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, Holder<Incomplete>*>);
static_assert(!cuda::std::is_invocable_v<RangeCREndT, Holder<Incomplete>*&>);
#endif // TEST_STD_VER > 2017

int main(int, char**)
{
  static_assert(testReturnTypes());

  testArray();
#ifndef TEST_COMPILER_CUDACC_BELOW_11_3
  static_assert(testArray());
#endif // TEST_COMPILER_CUDACC_BELOW_11_3

  testREndMember();
  static_assert(testREndMember());

  testREndFunction();
  static_assert(testREndFunction());

  testBeginEnd();
  static_assert(testBeginEnd());

#if !defined(TEST_COMPILER_MSVC_2019)
  unused(ntmre);
  unused(ntare);
#endif // !TEST_COMPILER_MSVC_2019
#if !defined(TEST_COMPILER_ICC)
  unused(ntmrerr);
#endif // !TEST_COMPILER_ICC
  unused(rerar);
  unused(ntbte);
#if !defined(TEST_COMPILER_ICC)
  unused(ntetb);
#endif // !TEST_COMPILER_ICC

  return 0;
}
