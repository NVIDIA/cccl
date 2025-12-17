//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// std::ranges::begin
// std::ranges::cbegin

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"

using RangeBeginT  = decltype(cuda::std::ranges::begin);
using RangeCBeginT = decltype(cuda::std::ranges::cbegin);

TEST_GLOBAL_VARIABLE int globalBuff[8] = {};

// This has been made valid as a defect report for C++17 onwards, however both clang and gcc below 11.0 does not
// implement it
#if !TEST_COMPILER(GCC, <, 11)
static_assert(cuda::std::is_invocable_v<RangeBeginT, int (&)[]>, "");
static_assert(cuda::std::is_invocable_v<RangeCBeginT, int (&)[]>, "");
#endif // !TEST_COMPILER(GCC, <, 11)
static_assert(cuda::std::is_invocable_v<RangeBeginT, int (&)[10]>, "");
static_assert(cuda::std::is_invocable_v<RangeCBeginT, int (&)[10]>, "");

#if !TEST_COMPILER(MSVC, <, 19, 23)
// old MSVC has a bug where it doesn't properly handle rvalue arrays
static_assert(!cuda::std::is_invocable_v<RangeBeginT, int (&&)[]>, "");
static_assert(!cuda::std::is_invocable_v<RangeBeginT, int (&&)[10]>, "");
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, int (&&)[]>, "");
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, int (&&)[10]>, "");
#endif // !TEST_COMPILER(MSVC, <, 19, 23)

struct Incomplete;
static_assert(!cuda::std::is_invocable_v<RangeBeginT, Incomplete (&&)[]>, "");
static_assert(!cuda::std::is_invocable_v<RangeBeginT, const Incomplete (&&)[]>, "");
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, Incomplete (&&)[]>, "");
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, const Incomplete (&&)[]>, "");

static_assert(!cuda::std::is_invocable_v<RangeBeginT, Incomplete (&&)[10]>, "");
static_assert(!cuda::std::is_invocable_v<RangeBeginT, const Incomplete (&&)[10]>, "");
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, Incomplete (&&)[10]>, "");
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, const Incomplete (&&)[10]>, "");

// This case is IFNDR; we handle it SFINAE-friendly.
static_assert(!cuda::std::is_invocable_v<RangeBeginT, Incomplete (&)[]>, "");
static_assert(!cuda::std::is_invocable_v<RangeBeginT, const Incomplete (&)[]>, "");
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, Incomplete (&)[]>, "");
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, const Incomplete (&)[]>, "");

// This case is IFNDR; we handle it SFINAE-friendly.
static_assert(!cuda::std::is_invocable_v<RangeBeginT, Incomplete (&)[10]>, "");
static_assert(!cuda::std::is_invocable_v<RangeBeginT, const Incomplete (&)[10]>, "");
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, Incomplete (&)[10]>, "");
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, const Incomplete (&)[10]>, "");

struct BeginMember
{
  int x;
  __host__ __device__ constexpr const int* begin() const
  {
    return &x;
  }
};

// Ensure that we can't call with rvalues with borrowing disabled.
static_assert(cuda::std::is_invocable_v<RangeBeginT, BeginMember&>, "");
static_assert(!cuda::std::is_invocable_v<RangeBeginT, BeginMember&&>, "");
static_assert(cuda::std::is_invocable_v<RangeBeginT, BeginMember const&>, "");
static_assert(!cuda::std::is_invocable_v<RangeBeginT, BeginMember const&&>, "");
static_assert(cuda::std::is_invocable_v<RangeCBeginT, BeginMember&>, "");
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, BeginMember&&>, "");
static_assert(cuda::std::is_invocable_v<RangeCBeginT, BeginMember const&>, "");
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, BeginMember const&&>, "");

struct Different
{
  __host__ __device__ char*& begin();
  __host__ __device__ short*& begin() const;
};
__host__ __device__ constexpr bool testReturnTypes()
{
  {
    int* x[2] = {};
    unused(x);
    static_assert(cuda::std::same_as<decltype(cuda::std::ranges::begin(x)), int**>, "");
    static_assert(cuda::std::same_as<decltype(cuda::std::ranges::cbegin(x)), int* const*>, "");
  }
  {
    int x[2][2] = {};
    unused(x);
    static_assert(cuda::std::same_as<decltype(cuda::std::ranges::begin(x)), int (*)[2]>, "");
    static_assert(cuda::std::same_as<decltype(cuda::std::ranges::cbegin(x)), const int (*)[2]>, "");
  }
  {
    Different x{};
    unused(x);
    static_assert(cuda::std::same_as<decltype(cuda::std::ranges::begin(x)), char*>, "");
    static_assert(cuda::std::same_as<decltype(cuda::std::ranges::cbegin(x)), short*>, "");
  }
  return true;
}

__host__ __device__ constexpr bool testArray()
{
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

struct BeginMemberReturnsInt
{
  __host__ __device__ int begin() const;
};
static_assert(!cuda::std::is_invocable_v<RangeBeginT, BeginMemberReturnsInt const&>, "");

struct BeginMemberReturnsVoidPtr
{
  __host__ __device__ const void* begin() const;
};
static_assert(!cuda::std::is_invocable_v<RangeBeginT, BeginMemberReturnsVoidPtr const&>, "");

struct EmptyBeginMember
{
  struct iterator
  {};
  __host__ __device__ iterator begin() const;
};
static_assert(!cuda::std::is_invocable_v<RangeBeginT, EmptyBeginMember const&>, "");

struct PtrConvertibleBeginMember
{
  struct iterator
  {
    __host__ __device__ operator int*() const;
  };
  __host__ __device__ iterator begin() const;
};
static_assert(!cuda::std::is_invocable_v<RangeBeginT, PtrConvertibleBeginMember const&>, "");

struct NonConstBeginMember
{
  int x;
  __host__ __device__ constexpr int* begin()
  {
    return &x;
  }
};
static_assert(cuda::std::is_invocable_v<RangeBeginT, NonConstBeginMember&>, "");
static_assert(!cuda::std::is_invocable_v<RangeBeginT, NonConstBeginMember const&>, "");
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, NonConstBeginMember&>, "");
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, NonConstBeginMember const&>, "");

struct EnabledBorrowingBeginMember
{
  __host__ __device__ constexpr const int* begin() const
  {
    return &globalBuff[0];
  }
};

namespace cuda::std::ranges
{
template <>
inline constexpr bool enable_borrowed_range<EnabledBorrowingBeginMember> = true;
}

struct BeginMemberFunction
{
  int x;
  __host__ __device__ constexpr const int* begin() const
  {
    return &x;
  }
  __host__ __device__ friend int* begin(BeginMemberFunction const&);
};

struct EmptyPtrBeginMember
{
  struct Empty
  {};
  Empty x;
  __host__ __device__ constexpr const Empty* begin() const
  {
    return &x;
  }
};

__host__ __device__ constexpr bool testBeginMember()
{
  BeginMember a{};
  assert(cuda::std::ranges::begin(a) == &a.x);
  assert(cuda::std::ranges::cbegin(a) == &a.x);
  static_assert(!cuda::std::is_invocable_v<RangeBeginT, BeginMember&&>, "");
  static_assert(!cuda::std::is_invocable_v<RangeCBeginT, BeginMember&&>, "");

  NonConstBeginMember b{};
  assert(cuda::std::ranges::begin(b) == &b.x);
  static_assert(!cuda::std::is_invocable_v<RangeCBeginT, NonConstBeginMember&>, "");

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

struct BeginFunction
{
  int x;
  __host__ __device__ friend constexpr const int* begin(BeginFunction const& bf)
  {
    return &bf.x;
  }
};
static_assert(cuda::std::is_invocable_v<RangeBeginT, BeginFunction const&>, "");
static_assert(!cuda::std::is_invocable_v<RangeBeginT, BeginFunction&&>, "");
static_assert(!cuda::std::is_invocable_v<RangeBeginT, BeginFunction&>, "");
static_assert(cuda::std::is_invocable_v<RangeCBeginT, BeginFunction const&>, "");
static_assert(cuda::std::is_invocable_v<RangeCBeginT, BeginFunction&>, "");

struct BeginFunctionReturnsInt
{
  __host__ __device__ friend int begin(BeginFunctionReturnsInt const&);
};
static_assert(!cuda::std::is_invocable_v<RangeBeginT, BeginFunctionReturnsInt const&>, "");

struct BeginFunctionReturnsVoidPtr
{
  __host__ __device__ friend void* begin(BeginFunctionReturnsVoidPtr const&);
};
static_assert(!cuda::std::is_invocable_v<RangeBeginT, BeginFunctionReturnsVoidPtr const&>, "");

struct BeginFunctionReturnsPtrConvertible
{
  struct iterator
  {
    __host__ __device__ operator int*() const;
  };
  __host__ __device__ friend iterator begin(BeginFunctionReturnsPtrConvertible const&);
};
static_assert(!cuda::std::is_invocable_v<RangeBeginT, BeginFunctionReturnsPtrConvertible const&>, "");

struct BeginFunctionByValue
{
  __host__ __device__ friend constexpr const int* begin(BeginFunctionByValue)
  {
    return &globalBuff[1];
  }
};
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, BeginFunctionByValue>, "");

struct BeginFunctionEnabledBorrowing
{
  __host__ __device__ friend constexpr const int* begin(BeginFunctionEnabledBorrowing)
  {
    return &globalBuff[2];
  }
};

namespace cuda::std::ranges
{
template <>
inline constexpr bool enable_borrowed_range<BeginFunctionEnabledBorrowing> = true;
}

struct BeginFunctionReturnsEmptyPtr
{
  struct Empty
  {};
  Empty x;
  __host__ __device__ friend constexpr const Empty* begin(BeginFunctionReturnsEmptyPtr const& bf)
  {
    return &bf.x;
  }
};

struct BeginFunctionWithDataMember
{
  int x;
  int begin;
  __host__ __device__ friend constexpr const int* begin(BeginFunctionWithDataMember const& bf)
  {
    return &bf.x;
  }
};

struct BeginFunctionWithPrivateBeginMember
{
  int y;
  __host__ __device__ friend constexpr const int* begin(BeginFunctionWithPrivateBeginMember const& bf)
  {
    return &bf.y;
  }

private:
  __host__ __device__ const int* begin() const;
};

__host__ __device__ constexpr bool testBeginFunction()
{
  BeginFunction a{};
  const BeginFunction aa{};
  static_assert(!cuda::std::invocable<RangeBeginT, decltype((a))>, "");
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
  static_assert(!cuda::std::invocable<RangeBeginT, decltype((d))>, "");
  assert(cuda::std::ranges::cbegin(d) == &d.x);
  assert(cuda::std::ranges::begin(dd) == &dd.x);
  assert(cuda::std::ranges::cbegin(dd) == &dd.x);

  BeginFunctionWithDataMember e{};
  const BeginFunctionWithDataMember ee{};
  static_assert(!cuda::std::invocable<RangeBeginT, decltype((e))>, "");
  assert(cuda::std::ranges::begin(ee) == &ee.x);
  assert(cuda::std::ranges::cbegin(e) == &e.x);
  assert(cuda::std::ranges::cbegin(ee) == &ee.x);

  BeginFunctionWithPrivateBeginMember f{};
  const BeginFunctionWithPrivateBeginMember ff{};
  static_assert(!cuda::std::invocable<RangeBeginT, decltype((f))>, "");
  assert(cuda::std::ranges::cbegin(f) == &f.y);
  assert(cuda::std::ranges::begin(ff) == &ff.y);
  assert(cuda::std::ranges::cbegin(ff) == &ff.y);

  return true;
}
static_assert(noexcept(cuda::std::ranges::begin(cuda::std::declval<int (&)[10]>())));
static_assert(noexcept(cuda::std::ranges::cbegin(cuda::std::declval<int (&)[10]>())));

// needs c++17's guaranteed copy elision
#if !TEST_COMPILER(MSVC2019) // broken noexcept
_CCCL_GLOBAL_CONSTANT struct NoThrowMemberBegin
{
  __host__ __device__ ThrowingIterator<int> begin() const noexcept; // auto(t.begin()) doesn't throw
} ntmb;
static_assert(noexcept(cuda::std::ranges::begin(ntmb)), "");
static_assert(noexcept(cuda::std::ranges::cbegin(ntmb)), "");

_CCCL_GLOBAL_CONSTANT struct NoThrowADLBegin
{
  __host__ __device__ friend ThrowingIterator<int> begin(NoThrowADLBegin&) noexcept; // auto(begin(t)) doesn't throw
  __host__ __device__ friend ThrowingIterator<int> begin(const NoThrowADLBegin&) noexcept;
} ntab;
static_assert(noexcept(cuda::std::ranges::begin(ntab)), "");
static_assert(noexcept(cuda::std::ranges::cbegin(ntab)), "");
#endif // !TEST_COMPILER(MSVC2019)

_CCCL_GLOBAL_CONSTANT struct NoThrowMemberBeginReturnsRef
{
  __host__ __device__ ThrowingIterator<int>& begin() const noexcept; // auto(t.begin()) may throw
} ntmbrr;
static_assert(!noexcept(cuda::std::ranges::begin(ntmbrr)), "");
static_assert(!noexcept(cuda::std::ranges::cbegin(ntmbrr)), "");

_CCCL_GLOBAL_CONSTANT struct BeginReturnsArrayRef
{
  __host__ __device__ auto begin() const noexcept -> int (&)[10];
} brar;
static_assert(noexcept(cuda::std::ranges::begin(brar)), "");
static_assert(noexcept(cuda::std::ranges::cbegin(brar)), "");

#if TEST_STD_VER > 2017
// Test ADL-proofing.
struct Incomplete;
template <class T>
struct Holder
{
  T t;
};
static_assert(!cuda::std::is_invocable_v<RangeBeginT, Holder<Incomplete>*>, "");
static_assert(!cuda::std::is_invocable_v<RangeBeginT, Holder<Incomplete>*&>, "");
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, Holder<Incomplete>*>, "");
static_assert(!cuda::std::is_invocable_v<RangeCBeginT, Holder<Incomplete>*&>, "");
#endif // TEST_STD_VER > 2017

int main(int, char**)
{
  static_assert(testReturnTypes(), "");

  testArray();
  static_assert(testArray(), "");

  testBeginMember();
  static_assert(testBeginMember(), "");

  testBeginFunction();
  static_assert(testBeginFunction(), "");

#if !TEST_COMPILER(MSVC2019) // broken noexcept
  unused(ntmb);
  unused(ntab);
#endif // !TEST_COMPILER(MSVC2019)
  unused(ntmbrr);
  unused(brar);

  return 0;
}
