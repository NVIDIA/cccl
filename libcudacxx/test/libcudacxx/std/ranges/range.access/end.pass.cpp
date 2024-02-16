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

// cuda::std::ranges::end
// cuda::std::ranges::cend

#include <cuda/std/ranges>

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"
#include "test_iterators.h"

using RangeEndT = decltype(cuda::std::ranges::end);
using RangeCEndT = decltype(cuda::std::ranges::cend);

STATIC_TEST_GLOBAL_VAR int globalBuff[8] = {};

static_assert(!cuda::std::is_invocable_v<RangeEndT, int (&&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeEndT, int (&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeEndT, int (&&)[10]>);
static_assert( cuda::std::is_invocable_v<RangeEndT, int (&)[10]>);
static_assert(!cuda::std::is_invocable_v<RangeCEndT, int (&&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeCEndT, int (&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeCEndT, int (&&)[10]>);
static_assert( cuda::std::is_invocable_v<RangeCEndT, int (&)[10]>);

struct Incomplete;
static_assert(!cuda::std::is_invocable_v<RangeEndT, Incomplete(&&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeEndT, Incomplete(&&)[42]>);
static_assert(!cuda::std::is_invocable_v<RangeCEndT, Incomplete(&&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeCEndT, Incomplete(&&)[42]>);

struct EndMember {
  int x;
  TEST_HOST_DEVICE const int *begin() const;
  TEST_HOST_DEVICE constexpr const int *end() const { return &x; }
};

// Ensure that we can't call with rvalues with borrowing disabled.
static_assert( cuda::std::is_invocable_v<RangeEndT, EndMember &>);
static_assert(!cuda::std::is_invocable_v<RangeEndT, EndMember &&>);
static_assert( cuda::std::is_invocable_v<RangeEndT, EndMember const&>);
static_assert(!cuda::std::is_invocable_v<RangeEndT, EndMember const&&>);
static_assert( cuda::std::is_invocable_v<RangeCEndT, EndMember &>);
static_assert(!cuda::std::is_invocable_v<RangeCEndT, EndMember &&>);
static_assert( cuda::std::is_invocable_v<RangeCEndT, EndMember const&>);
static_assert(!cuda::std::is_invocable_v<RangeCEndT, EndMember const&&>);

struct Different {
  TEST_HOST_DEVICE char *begin();
  TEST_HOST_DEVICE sentinel_wrapper<char*>& end();
  TEST_HOST_DEVICE short *begin() const;
  TEST_HOST_DEVICE sentinel_wrapper<short*>& end() const;
};
TEST_HOST_DEVICE constexpr bool testReturnTypes() {
  {
    int *x[2] = {};
    unused(x);
    static_assert(cuda::std::same_as<decltype(cuda::std::ranges::end(x)), int**>);
    static_assert(cuda::std::same_as<decltype(cuda::std::ranges::cend(x)), int* const*>);
  }
  {
    int x[2][2] = {};
    unused(x);
    static_assert(cuda::std::same_as<decltype(cuda::std::ranges::end(x)), int(*)[2]>);
    static_assert(cuda::std::same_as<decltype(cuda::std::ranges::cend(x)), const int(*)[2]>);
  }
  {
    Different x{};
    unused(x);
    static_assert(cuda::std::same_as<decltype(cuda::std::ranges::end(x)), sentinel_wrapper<char*>>);
    static_assert(cuda::std::same_as<decltype(cuda::std::ranges::cend(x)), sentinel_wrapper<short*>>);
  }
  return true;
}

TEST_HOST_DEVICE constexpr bool testArray() {
  int a[2] = {};
  assert(cuda::std::ranges::end(a) == a + 2);
  assert(cuda::std::ranges::cend(a) == a + 2);

  int b[2][2] = {};
  assert(cuda::std::ranges::end(b) == b + 2);
  assert(cuda::std::ranges::cend(b) == b + 2);

  EndMember c[2] = {};
  assert(cuda::std::ranges::end(c) == c + 2);
  assert(cuda::std::ranges::cend(c) == c + 2);

  return true;
}

struct EndMemberReturnsInt {
  TEST_HOST_DEVICE int begin() const;
  TEST_HOST_DEVICE int end() const;
};
static_assert(!cuda::std::is_invocable_v<RangeEndT, EndMemberReturnsInt const&>);

struct EndMemberReturnsVoidPtr {
  TEST_HOST_DEVICE const void *begin() const;
  TEST_HOST_DEVICE const void *end() const;
};
static_assert(!cuda::std::is_invocable_v<RangeEndT, EndMemberReturnsVoidPtr const&>);

struct PtrConvertible {
  TEST_HOST_DEVICE operator int*() const;
};
struct PtrConvertibleEndMember {
  TEST_HOST_DEVICE PtrConvertible begin() const;
  TEST_HOST_DEVICE PtrConvertible end() const;
};
static_assert(!cuda::std::is_invocable_v<RangeEndT, PtrConvertibleEndMember const&>);

struct NoBeginMember {
  TEST_HOST_DEVICE constexpr const int *end();
};
static_assert(!cuda::std::is_invocable_v<RangeEndT, NoBeginMember const&>);

struct NonConstEndMember {
  int x;
  TEST_HOST_DEVICE constexpr int *begin() { return nullptr; }
  TEST_HOST_DEVICE constexpr int *end() { return &x; }
};
static_assert( cuda::std::is_invocable_v<RangeEndT,  NonConstEndMember &>);
static_assert(!cuda::std::is_invocable_v<RangeEndT,  NonConstEndMember const&>);
static_assert(!cuda::std::is_invocable_v<RangeCEndT, NonConstEndMember &>);
static_assert(!cuda::std::is_invocable_v<RangeCEndT, NonConstEndMember const&>);

struct EnabledBorrowingEndMember {
  TEST_HOST_DEVICE constexpr int *begin() const { return nullptr; }
  TEST_HOST_DEVICE constexpr int *end() const { return &globalBuff[0]; }
};
template<>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<EnabledBorrowingEndMember> = true;

struct EndMemberFunction {
  int x;
  TEST_HOST_DEVICE constexpr const int *begin() const { return nullptr; }
  TEST_HOST_DEVICE constexpr const int *end() const { return &x; }
  TEST_HOST_DEVICE friend constexpr int *end(EndMemberFunction const&);
};

struct Empty { };
struct EmptyEndMember {
  TEST_HOST_DEVICE Empty begin() const;
  TEST_HOST_DEVICE Empty end() const;
};
static_assert(!cuda::std::is_invocable_v<RangeEndT, EmptyEndMember const&>);

struct EmptyPtrEndMember {
  Empty x;
  TEST_HOST_DEVICE constexpr const Empty *begin() const { return nullptr; }
  TEST_HOST_DEVICE constexpr const Empty *end() const { return &x; }
};

TEST_HOST_DEVICE constexpr bool testEndMember() {
  EndMember a{};
  assert(cuda::std::ranges::end(a) == &a.x);
  assert(cuda::std::ranges::cend(a) == &a.x);

  NonConstEndMember b{};
  assert(cuda::std::ranges::end(b) == &b.x);
  static_assert(!cuda::std::is_invocable_v<RangeCEndT, decltype((b))>);

  EnabledBorrowingEndMember c{};
  assert(cuda::std::ranges::end(cuda::std::move(c)) == &globalBuff[0]);
  assert(cuda::std::ranges::cend(cuda::std::move(c)) == &globalBuff[0]);

  EndMemberFunction d{};
  assert(cuda::std::ranges::end(d) == &d.x);
  assert(cuda::std::ranges::cend(d) == &d.x);

  EmptyPtrEndMember e{};
  assert(cuda::std::ranges::end(e) == &e.x);
  assert(cuda::std::ranges::cend(e) == &e.x);

  return true;
}

struct EndFunction {
  int x;
  TEST_HOST_DEVICE friend constexpr const int *begin(EndFunction const&) { return nullptr; }
  TEST_HOST_DEVICE friend constexpr const int *end(EndFunction const& bf) { return &bf.x; }
};

static_assert( cuda::std::is_invocable_v<RangeEndT, EndFunction const&>);
static_assert(!cuda::std::is_invocable_v<RangeEndT, EndFunction &&>);

static_assert( cuda::std::is_invocable_v<RangeEndT,  EndFunction const&>);
static_assert(!cuda::std::is_invocable_v<RangeEndT,  EndFunction &&>);
static_assert(!cuda::std::is_invocable_v<RangeEndT,  EndFunction &>);
static_assert( cuda::std::is_invocable_v<RangeCEndT, EndFunction const&>);
static_assert( cuda::std::is_invocable_v<RangeCEndT, EndFunction &>);

struct EndFunctionReturnsInt {
  TEST_HOST_DEVICE friend constexpr int begin(EndFunctionReturnsInt const&);
  TEST_HOST_DEVICE friend constexpr int end(EndFunctionReturnsInt const&);
};
static_assert(!cuda::std::is_invocable_v<RangeEndT, EndFunctionReturnsInt const&>);

struct EndFunctionReturnsVoidPtr {
  TEST_HOST_DEVICE friend constexpr void *begin(EndFunctionReturnsVoidPtr const&);
  TEST_HOST_DEVICE friend constexpr void *end(EndFunctionReturnsVoidPtr const&);
};
static_assert(!cuda::std::is_invocable_v<RangeEndT, EndFunctionReturnsVoidPtr const&>);

struct EndFunctionReturnsEmpty {
  TEST_HOST_DEVICE friend constexpr Empty begin(EndFunctionReturnsEmpty const&);
  TEST_HOST_DEVICE friend constexpr Empty end(EndFunctionReturnsEmpty const&);
};
static_assert(!cuda::std::is_invocable_v<RangeEndT, EndFunctionReturnsEmpty const&>);

struct EndFunctionReturnsPtrConvertible {
  TEST_HOST_DEVICE friend constexpr PtrConvertible begin(EndFunctionReturnsPtrConvertible const&);
  TEST_HOST_DEVICE friend constexpr PtrConvertible end(EndFunctionReturnsPtrConvertible const&);
};
static_assert(!cuda::std::is_invocable_v<RangeEndT, EndFunctionReturnsPtrConvertible const&>);

struct NoBeginFunction {
  TEST_HOST_DEVICE friend constexpr const int *end(NoBeginFunction const&);
};
static_assert(!cuda::std::is_invocable_v<RangeEndT, NoBeginFunction const&>);

struct EndFunctionByValue {
  TEST_HOST_DEVICE friend constexpr int *begin(EndFunctionByValue) { return nullptr; }
  TEST_HOST_DEVICE friend constexpr int *end(EndFunctionByValue) { return &globalBuff[1]; }
};
static_assert(!cuda::std::is_invocable_v<RangeCEndT, EndFunctionByValue>);

struct EndFunctionEnabledBorrowing {
  TEST_HOST_DEVICE friend constexpr int *begin(EndFunctionEnabledBorrowing) { return nullptr; }
  TEST_HOST_DEVICE friend constexpr int *end(EndFunctionEnabledBorrowing) { return &globalBuff[2]; }
};
template<>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<EndFunctionEnabledBorrowing> = true;

struct EndFunctionReturnsEmptyPtr {
  Empty x;
  TEST_HOST_DEVICE friend constexpr const Empty *begin(EndFunctionReturnsEmptyPtr const&) { return nullptr; }
  TEST_HOST_DEVICE friend constexpr const Empty *end(EndFunctionReturnsEmptyPtr const& bf) { return &bf.x; }
};

struct EndFunctionWithDataMember {
  int x;
  int end;
  TEST_HOST_DEVICE friend constexpr const int *begin(EndFunctionWithDataMember const&) { return nullptr; }
  TEST_HOST_DEVICE friend constexpr const int *end(EndFunctionWithDataMember const& bf) { return &bf.x; }
};

struct EndFunctionWithPrivateEndMember {
  int y;
  TEST_HOST_DEVICE friend constexpr const int *begin(EndFunctionWithPrivateEndMember const&) { return nullptr; }
  TEST_HOST_DEVICE friend constexpr const int *end(EndFunctionWithPrivateEndMember const& bf) { return &bf.y; }
private:
  TEST_HOST_DEVICE const int *end() const;
};

struct BeginMemberEndFunction {
  int x;
  TEST_HOST_DEVICE constexpr const int *begin() const { return nullptr; }
  TEST_HOST_DEVICE friend constexpr const int *end(BeginMemberEndFunction const& bf) { return &bf.x; }
};

TEST_HOST_DEVICE constexpr bool testEndFunction() {
  const EndFunction a{};
  assert(cuda::std::ranges::end(a) == &a.x);
  assert(cuda::std::ranges::cend(a) == &a.x);
  EndFunction aa{};
  static_assert(!cuda::std::is_invocable_v<RangeEndT, decltype((aa))>);
  assert(cuda::std::ranges::cend(aa) == &aa.x);

  EndFunctionByValue b{};
  assert(cuda::std::ranges::end(b) == &globalBuff[1]);
  assert(cuda::std::ranges::cend(b) == &globalBuff[1]);

  EndFunctionEnabledBorrowing c{};
  assert(cuda::std::ranges::end(cuda::std::move(c)) == &globalBuff[2]);
  assert(cuda::std::ranges::cend(cuda::std::move(c)) == &globalBuff[2]);

  const EndFunctionReturnsEmptyPtr d{};
  assert(cuda::std::ranges::end(d) == &d.x);
  assert(cuda::std::ranges::cend(d) == &d.x);
  EndFunctionReturnsEmptyPtr dd{};
  static_assert(!cuda::std::is_invocable_v<RangeEndT, decltype((dd))>);
  assert(cuda::std::ranges::cend(dd) == &dd.x);

  const EndFunctionWithDataMember e{};
  assert(cuda::std::ranges::end(e) == &e.x);
  assert(cuda::std::ranges::cend(e) == &e.x);
  EndFunctionWithDataMember ee{};
  static_assert(!cuda::std::is_invocable_v<RangeEndT, decltype((ee))>);
  assert(cuda::std::ranges::cend(ee) == &ee.x);

  const EndFunctionWithPrivateEndMember f{};
  assert(cuda::std::ranges::end(f) == &f.y);
  assert(cuda::std::ranges::cend(f) == &f.y);
  EndFunctionWithPrivateEndMember ff{};
  static_assert(!cuda::std::is_invocable_v<RangeEndT, decltype((ff))>);
  assert(cuda::std::ranges::cend(ff) == &ff.y);

  const BeginMemberEndFunction g{};
  assert(cuda::std::ranges::end(g) == &g.x);
  assert(cuda::std::ranges::cend(g) == &g.x);
  BeginMemberEndFunction gg{};
  static_assert(!cuda::std::is_invocable_v<RangeEndT, decltype((gg))>);
  assert(cuda::std::ranges::cend(gg) == &gg.x);

  return true;
}
ASSERT_NOEXCEPT(cuda::std::ranges::end(cuda::std::declval<int (&)[10]>()));
ASSERT_NOEXCEPT(cuda::std::ranges::cend(cuda::std::declval<int (&)[10]>()));

#if !defined(TEST_COMPILER_MSVC_2019) // broken noexcept
_LIBCUDACXX_CPO_ACCESSIBILITY struct NoThrowMemberEnd {
  TEST_HOST_DEVICE ThrowingIterator<int> begin() const;
  TEST_HOST_DEVICE ThrowingIterator<int> end() const noexcept; // auto(t.end()) doesn't throw
} ntme;
static_assert(noexcept(cuda::std::ranges::end(ntme)));
static_assert(noexcept(cuda::std::ranges::cend(ntme)));

_LIBCUDACXX_CPO_ACCESSIBILITY struct NoThrowADLEnd {
  TEST_HOST_DEVICE ThrowingIterator<int> begin() const;
  TEST_HOST_DEVICE friend ThrowingIterator<int> end(NoThrowADLEnd&) noexcept;  // auto(end(t)) doesn't throw
  TEST_HOST_DEVICE friend ThrowingIterator<int> end(const NoThrowADLEnd&) noexcept;
} ntae;
static_assert(noexcept(cuda::std::ranges::end(ntae)));
static_assert(noexcept(cuda::std::ranges::cend(ntae)));
#endif // !TEST_COMPILER_MSVC_2019

#if !defined(TEST_COMPILER_ICC)
_LIBCUDACXX_CPO_ACCESSIBILITY struct NoThrowMemberEndReturnsRef {
  TEST_HOST_DEVICE ThrowingIterator<int> begin() const;
  TEST_HOST_DEVICE ThrowingIterator<int>& end() const noexcept; // auto(t.end()) may throw
} ntmerr;
static_assert(!noexcept(cuda::std::ranges::end(ntmerr)));
static_assert(!noexcept(cuda::std::ranges::cend(ntmerr)));
#endif // !TEST_COMPILER_ICC

_LIBCUDACXX_CPO_ACCESSIBILITY struct EndReturnsArrayRef {
  TEST_HOST_DEVICE auto begin() const noexcept -> int(&)[10];
  TEST_HOST_DEVICE auto end() const noexcept -> int(&)[10];
} erar;
static_assert(noexcept(cuda::std::ranges::end(erar)));
static_assert(noexcept(cuda::std::ranges::cend(erar)));

#if TEST_STD_VER > 2017
// Test ADL-proofing.
struct Incomplete;
template<class T> struct Holder { T t; };
static_assert(!cuda::std::is_invocable_v<RangeEndT, Holder<Incomplete>*>);
static_assert(!cuda::std::is_invocable_v<RangeEndT, Holder<Incomplete>*&>);
static_assert(!cuda::std::is_invocable_v<RangeCEndT, Holder<Incomplete>*>);
static_assert(!cuda::std::is_invocable_v<RangeCEndT, Holder<Incomplete>*&>);
#endif // TEST_STD_VER > 2017

int main(int, char**) {
  static_assert(testReturnTypes());

  testArray();
#ifndef TEST_COMPILER_CUDACC_BELOW_11_3
  static_assert(testArray());
#endif // TEST_COMPILER_CUDACC_BELOW_11_3

  testEndMember();
  static_assert(testEndMember());

  testEndFunction();
  static_assert(testEndFunction());

#if !defined(TEST_COMPILER_MSVC_2019) // broken noexcept
  unused(ntme);
  unused(ntae);
#endif // !TEST_COMPILER_MSVC_2019
#if !defined(TEST_COMPILER_ICC)
  unused(ntmerr);
#endif // !TEST_COMPILER_ICC
  unused(erar);

  return 0;
}
