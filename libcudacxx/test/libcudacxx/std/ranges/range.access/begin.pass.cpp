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

// std::ranges::begin
// std::ranges::cbegin

#include <cuda/std/ranges>

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"
#include "test_iterators.h"

using RangeBeginT = decltype(cuda::std::ranges::begin);
using RangeCBeginT = decltype(cuda::std::ranges::cbegin);

STATIC_TEST_GLOBAL_VAR int globalBuff[8] = {};

static_assert(!cuda::std::is_invocable_v<RangeBeginT, int (&&)[10]>);
static_assert( cuda::std::is_invocable_v<RangeBeginT, int (&)[10]>);
static_assert(!cuda::std::is_invocable_v<RangeBeginT, int (&&)[]>);

// This has been made valid as a defect report for C++17 onwards, however both clang and gcc below 11.0 does not implement it
#if (!defined(__GNUC__) || __GNUC__ >= 11)
static_assert( cuda::std::is_invocable_v<RangeBeginT, int (&)[]>);
#endif
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, int (&&)[10]>);
static_assert( cuda::std::is_invocable_v<RangeCBeginT, int (&)[10]>);
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, int (&&)[]>);
// This has been made valid as a defect report for C++17 onwards, however both clang and gcc below 11.0 does not implement it
#if (!defined(__GNUC__) || __GNUC__ >= 11)
static_assert( cuda::std::is_invocable_v<RangeCBeginT, int (&)[]>);
#endif

struct Incomplete;
static_assert(!cuda::std::is_invocable_v<RangeBeginT, Incomplete(&&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeBeginT, const Incomplete(&&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, Incomplete(&&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, const Incomplete(&&)[]>);

static_assert(!cuda::std::is_invocable_v<RangeBeginT, Incomplete(&&)[10]>);
static_assert(!cuda::std::is_invocable_v<RangeBeginT, const Incomplete(&&)[10]>);
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, Incomplete(&&)[10]>);
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, const Incomplete(&&)[10]>);

// This case is IFNDR; we handle it SFINAE-friendly.
static_assert(!cuda::std::is_invocable_v<RangeBeginT, Incomplete(&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeBeginT, const Incomplete(&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, Incomplete(&)[]>);
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, const Incomplete(&)[]>);

// This case is IFNDR; we handle it SFINAE-friendly.
static_assert(!cuda::std::is_invocable_v<RangeBeginT, Incomplete(&)[10]>);
static_assert(!cuda::std::is_invocable_v<RangeBeginT, const Incomplete(&)[10]>);
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, Incomplete(&)[10]>);
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, const Incomplete(&)[10]>);

struct BeginMember {
  int x;
  TEST_HOST_DEVICE constexpr const int *begin() const { return &x; }
};

// Ensure that we can't call with rvalues with borrowing disabled.
static_assert( cuda::std::is_invocable_v<RangeBeginT, BeginMember &>);
static_assert(!cuda::std::is_invocable_v<RangeBeginT, BeginMember &&>);
static_assert( cuda::std::is_invocable_v<RangeBeginT, BeginMember const&>);
static_assert(!cuda::std::is_invocable_v<RangeBeginT, BeginMember const&&>);
static_assert( cuda::std::is_invocable_v<RangeCBeginT, BeginMember &>);
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, BeginMember &&>);
static_assert( cuda::std::is_invocable_v<RangeCBeginT, BeginMember const&>);
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, BeginMember const&&>);

struct Different {
  TEST_HOST_DEVICE char*& begin();
  TEST_HOST_DEVICE short*& begin() const;
};
TEST_HOST_DEVICE constexpr bool testReturnTypes() {
  {
    int *x[2] = {};
    unused(x);
    static_assert(cuda::std::same_as<decltype(cuda::std::ranges::begin(x)), int**>);
    static_assert(cuda::std::same_as<decltype(cuda::std::ranges::cbegin(x)), int* const*>);
  }
  {
    int x[2][2] = {};
    unused(x);
    static_assert(cuda::std::same_as<decltype(cuda::std::ranges::begin(x)), int(*)[2]>);
    static_assert(cuda::std::same_as<decltype(cuda::std::ranges::cbegin(x)), const int(*)[2]>);
  }
  {
    Different x{};
    unused(x);
    static_assert(cuda::std::same_as<decltype(cuda::std::ranges::begin(x)), char*>);
    static_assert(cuda::std::same_as<decltype(cuda::std::ranges::cbegin(x)), short*>);
  }
  return true;
}

TEST_HOST_DEVICE constexpr bool testArray() {
  int a[2] = {};
  assert(cuda::std::ranges::begin(a) == a);
  assert(cuda::std::ranges::cbegin(a) == a);

  int b[2][2] = {};
  assert(cuda::std::ranges::begin(b) == b);
  assert(cuda::std::ranges::cbegin(b) == b);

  BeginMember c[2] = {};
  assert(cuda::std::ranges::begin(c) == c);
  assert(cuda::std::ranges::cbegin(c) == c);

  return true;
}

struct BeginMemberReturnsInt {
  TEST_HOST_DEVICE int begin() const;
};
static_assert(!cuda::std::is_invocable_v<RangeBeginT, BeginMemberReturnsInt const&>);

struct BeginMemberReturnsVoidPtr {
  TEST_HOST_DEVICE const void *begin() const;
};
static_assert(!cuda::std::is_invocable_v<RangeBeginT, BeginMemberReturnsVoidPtr const&>);

struct EmptyBeginMember {
  struct iterator {};
  TEST_HOST_DEVICE iterator begin() const;
};
static_assert(!cuda::std::is_invocable_v<RangeBeginT, EmptyBeginMember const&>);

struct PtrConvertibleBeginMember {
  struct iterator { TEST_HOST_DEVICE operator int*() const; };
  TEST_HOST_DEVICE iterator begin() const;
};
static_assert(!cuda::std::is_invocable_v<RangeBeginT, PtrConvertibleBeginMember const&>);

struct NonConstBeginMember {
  int x;
  TEST_HOST_DEVICE constexpr int *begin() { return &x; }
};
static_assert( cuda::std::is_invocable_v<RangeBeginT,  NonConstBeginMember &>);
static_assert(!cuda::std::is_invocable_v<RangeBeginT,  NonConstBeginMember const&>);
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, NonConstBeginMember &>);
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, NonConstBeginMember const&>);

struct EnabledBorrowingBeginMember {
  TEST_HOST_DEVICE constexpr const int *begin() const { return &globalBuff[0]; }
};
template<>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<EnabledBorrowingBeginMember> = true;

struct BeginMemberFunction {
  int x;
  TEST_HOST_DEVICE constexpr const int *begin() const { return &x; }
  TEST_HOST_DEVICE friend int *begin(BeginMemberFunction const&);
};

struct EmptyPtrBeginMember {
  struct Empty {};
  Empty x;
  TEST_HOST_DEVICE constexpr const Empty *begin() const { return &x; }
};

TEST_HOST_DEVICE constexpr bool testBeginMember() {
  BeginMember a{};
  assert(cuda::std::ranges::begin(a) == &a.x);
  assert(cuda::std::ranges::cbegin(a) == &a.x);
  static_assert(!cuda::std::is_invocable_v<RangeBeginT, BeginMember&&>);
  static_assert(!cuda::std::is_invocable_v<RangeCBeginT, BeginMember&&>);

  NonConstBeginMember b{};
  assert(cuda::std::ranges::begin(b) == &b.x);
  static_assert(!cuda::std::is_invocable_v<RangeCBeginT, NonConstBeginMember&>);

  EnabledBorrowingBeginMember c{};
  assert(cuda::std::ranges::begin(c) == &globalBuff[0]);
  assert(cuda::std::ranges::cbegin(c) == &globalBuff[0]);
  assert(cuda::std::ranges::begin(cuda::std::move(c)) == &globalBuff[0]);
  assert(cuda::std::ranges::cbegin(cuda::std::move(c)) == &globalBuff[0]);

  BeginMemberFunction d{};
  assert(cuda::std::ranges::begin(d) == &d.x);
  assert(cuda::std::ranges::cbegin(d) == &d.x);

  EmptyPtrBeginMember e{};
  assert(cuda::std::ranges::begin(e) == &e.x);
  assert(cuda::std::ranges::cbegin(e) == &e.x);

  return true;
}


struct BeginFunction {
  int x;
  TEST_HOST_DEVICE friend constexpr const int *begin(BeginFunction const& bf) { return &bf.x; }
};
static_assert( cuda::std::is_invocable_v<RangeBeginT,  BeginFunction const&>);
static_assert(!cuda::std::is_invocable_v<RangeBeginT,  BeginFunction &&>);
static_assert(!cuda::std::is_invocable_v<RangeBeginT,  BeginFunction &>);
static_assert( cuda::std::is_invocable_v<RangeCBeginT, BeginFunction const&>);
static_assert( cuda::std::is_invocable_v<RangeCBeginT, BeginFunction &>);

struct BeginFunctionReturnsInt {
  TEST_HOST_DEVICE friend int begin(BeginFunctionReturnsInt const&);
};
static_assert(!cuda::std::is_invocable_v<RangeBeginT, BeginFunctionReturnsInt const&>);

struct BeginFunctionReturnsVoidPtr {
  TEST_HOST_DEVICE friend void *begin(BeginFunctionReturnsVoidPtr const&);
};
static_assert(!cuda::std::is_invocable_v<RangeBeginT, BeginFunctionReturnsVoidPtr const&>);

struct BeginFunctionReturnsPtrConvertible {
  struct iterator { TEST_HOST_DEVICE operator int*() const; };
  TEST_HOST_DEVICE friend iterator begin(BeginFunctionReturnsPtrConvertible const&);
};
static_assert(!cuda::std::is_invocable_v<RangeBeginT, BeginFunctionReturnsPtrConvertible const&>);

struct BeginFunctionByValue {
  TEST_HOST_DEVICE friend constexpr const int *begin(BeginFunctionByValue) { return &globalBuff[1]; }
};
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, BeginFunctionByValue>);

struct BeginFunctionEnabledBorrowing {
  TEST_HOST_DEVICE friend constexpr const int *begin(BeginFunctionEnabledBorrowing) { return &globalBuff[2]; }
};
template<>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<BeginFunctionEnabledBorrowing> = true;

struct BeginFunctionReturnsEmptyPtr {
  struct Empty {};
  Empty x;
  TEST_HOST_DEVICE friend constexpr const Empty *begin(BeginFunctionReturnsEmptyPtr const& bf) { return &bf.x; }
};

struct BeginFunctionWithDataMember {
  int x;
  int begin;
  TEST_HOST_DEVICE friend constexpr const int *begin(BeginFunctionWithDataMember const& bf) { return &bf.x; }
};

struct BeginFunctionWithPrivateBeginMember {
  int y;
  TEST_HOST_DEVICE friend constexpr const int *begin(BeginFunctionWithPrivateBeginMember const& bf) { return &bf.y; }
private:
  TEST_HOST_DEVICE const int *begin() const;
};

TEST_HOST_DEVICE constexpr bool testBeginFunction() {
  BeginFunction a{};
  const BeginFunction aa{};
  static_assert(!cuda::std::invocable<RangeBeginT, decltype((a))>);
  assert(cuda::std::ranges::cbegin(a) == &a.x);
  assert(cuda::std::ranges::begin(aa) == &aa.x);
  assert(cuda::std::ranges::cbegin(aa) == &aa.x);

  BeginFunctionByValue b{};
  const BeginFunctionByValue bb{};
  assert(cuda::std::ranges::begin(b) == &globalBuff[1]);
  assert(cuda::std::ranges::cbegin(b) == &globalBuff[1]);
  assert(cuda::std::ranges::begin(bb) == &globalBuff[1]);
  assert(cuda::std::ranges::cbegin(bb) == &globalBuff[1]);

  BeginFunctionEnabledBorrowing c{};
  const BeginFunctionEnabledBorrowing cc{};
  assert(cuda::std::ranges::begin(cuda::std::move(c)) == &globalBuff[2]);
  assert(cuda::std::ranges::cbegin(cuda::std::move(c)) == &globalBuff[2]);
  assert(cuda::std::ranges::begin(cuda::std::move(cc)) == &globalBuff[2]);
  assert(cuda::std::ranges::cbegin(cuda::std::move(cc)) == &globalBuff[2]);

  BeginFunctionReturnsEmptyPtr d{};
  const BeginFunctionReturnsEmptyPtr dd{};
  static_assert(!cuda::std::invocable<RangeBeginT, decltype((d))>);
  assert(cuda::std::ranges::cbegin(d) == &d.x);
  assert(cuda::std::ranges::begin(dd) == &dd.x);
  assert(cuda::std::ranges::cbegin(dd) == &dd.x);

  BeginFunctionWithDataMember e{};
  const BeginFunctionWithDataMember ee{};
  static_assert(!cuda::std::invocable<RangeBeginT, decltype((e))>);
  assert(cuda::std::ranges::begin(ee) == &ee.x);
  assert(cuda::std::ranges::cbegin(e) == &e.x);
  assert(cuda::std::ranges::cbegin(ee) == &ee.x);

  BeginFunctionWithPrivateBeginMember f{};
  const BeginFunctionWithPrivateBeginMember ff{};
  static_assert(!cuda::std::invocable<RangeBeginT, decltype((f))>);
  assert(cuda::std::ranges::cbegin(f) == &f.y);
  assert(cuda::std::ranges::begin(ff) == &ff.y);
  assert(cuda::std::ranges::cbegin(ff) == &ff.y);

  return true;
}
ASSERT_NOEXCEPT(cuda::std::ranges::begin(cuda::std::declval<int (&)[10]>()));
ASSERT_NOEXCEPT(cuda::std::ranges::cbegin(cuda::std::declval<int (&)[10]>()));

#if !defined(TEST_COMPILER_MSVC_2019) // broken noexcept
_LIBCUDACXX_CPO_ACCESSIBILITY struct NoThrowMemberBegin {
  TEST_HOST_DEVICE ThrowingIterator<int> begin() const noexcept; // auto(t.begin()) doesn't throw
} ntmb;
static_assert(noexcept(cuda::std::ranges::begin(ntmb)));
static_assert(noexcept(cuda::std::ranges::cbegin(ntmb)));

_LIBCUDACXX_CPO_ACCESSIBILITY struct NoThrowADLBegin {
  TEST_HOST_DEVICE friend ThrowingIterator<int> begin(NoThrowADLBegin&) noexcept;  // auto(begin(t)) doesn't throw
  TEST_HOST_DEVICE friend ThrowingIterator<int> begin(const NoThrowADLBegin&) noexcept;
} ntab;
static_assert(noexcept(cuda::std::ranges::begin(ntab)));
static_assert(noexcept(cuda::std::ranges::cbegin(ntab)));
#endif // !TEST_COMPILER_MSVC_2019

#if !defined(TEST_COMPILER_ICC)
_LIBCUDACXX_CPO_ACCESSIBILITY struct NoThrowMemberBeginReturnsRef {
  TEST_HOST_DEVICE ThrowingIterator<int>& begin() const noexcept; // auto(t.begin()) may throw
} ntmbrr;
static_assert(!noexcept(cuda::std::ranges::begin(ntmbrr)));
static_assert(!noexcept(cuda::std::ranges::cbegin(ntmbrr)));
#endif // !TEST_COMPILER_ICC

_LIBCUDACXX_CPO_ACCESSIBILITY struct BeginReturnsArrayRef {
  TEST_HOST_DEVICE auto begin() const noexcept -> int(&)[10];
} brar;
static_assert(noexcept(cuda::std::ranges::begin(brar)));
static_assert(noexcept(cuda::std::ranges::cbegin(brar)));

#if TEST_STD_VER > 2017
// Test ADL-proofing.
struct Incomplete;
template<class T> struct Holder { T t; };
static_assert(!cuda::std::is_invocable_v<RangeBeginT, Holder<Incomplete>*>);
static_assert(!cuda::std::is_invocable_v<RangeBeginT, Holder<Incomplete>*&>);
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, Holder<Incomplete>*>);
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, Holder<Incomplete>*&>);
#endif // TEST_STD_VER > 2017

int main(int, char**) {
  static_assert(testReturnTypes());

  testArray();
#ifndef TEST_COMPILER_CUDACC_BELOW_11_3
  static_assert(testArray());
#endif // TEST_COMPILER_CUDACC_BELOW_11_3

  testBeginMember();
  static_assert(testBeginMember());

  testBeginFunction();
  static_assert(testBeginFunction());

#if !defined(TEST_COMPILER_MSVC_2019) // broken noexcept
  unused(ntmb);
  unused(ntab);
#endif // !TEST_COMPILER_MSVC_2019
#if !defined(TEST_COMPILER_ICC)
  unused(ntmbrr);
#endif // !TEST_COMPILER_ICC
  unused(brar);

  return 0;
}
