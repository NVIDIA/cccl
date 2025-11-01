//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// cuda::std::ranges::size

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"

using RangeSizeT = decltype(cuda::std::ranges::size);

static_assert(!cuda::std::is_invocable_v<RangeSizeT, int[]>, "");
static_assert(cuda::std::is_invocable_v<RangeSizeT, int[1]>, "");
static_assert(cuda::std::is_invocable_v<RangeSizeT, int (&&)[1]>, "");
static_assert(cuda::std::is_invocable_v<RangeSizeT, int (&)[1]>, "");

struct Incomplete;
static_assert(!cuda::std::is_invocable_v<RangeSizeT, Incomplete[]>, "");
static_assert(!cuda::std::is_invocable_v<RangeSizeT, Incomplete (&)[]>, "");
static_assert(!cuda::std::is_invocable_v<RangeSizeT, Incomplete (&&)[]>, "");

#if !TEST_COMPILER(NVRTC)
extern Incomplete array_of_incomplete[42];
static_assert(cuda::std::ranges::size(array_of_incomplete) == 42, "");
static_assert(cuda::std::ranges::size(cuda::std::move(array_of_incomplete)) == 42, "");

extern const Incomplete const_array_of_incomplete[42];
static_assert(cuda::std::ranges::size(const_array_of_incomplete) == 42, "");
static_assert(cuda::std::ranges::size(static_cast<const Incomplete (&&)[42]>(array_of_incomplete)) == 42, "");
#endif // !TEST_COMPILER(NVRTC)

struct SizeMember
{
  __host__ __device__ constexpr size_t size()
  {
    return 42;
  }
};

struct StaticSizeMember
{
  __host__ __device__ constexpr static size_t size()
  {
    return 42;
  }
};

static_assert(!cuda::std::is_invocable_v<RangeSizeT, const SizeMember>, "");

struct SizeFunction
{
  __host__ __device__ friend constexpr size_t size(SizeFunction)
  {
    return 42;
  }
};

// Make sure the size member is preferred.
struct SizeMemberAndFunction
{
  __host__ __device__ constexpr size_t size()
  {
    return 42;
  }
  __host__ __device__ friend constexpr size_t size(SizeMemberAndFunction)
  {
    return 0;
  }
};

__host__ __device__ bool constexpr testArrayType()
{
  int a[4]          = {};
  int b[1]          = {};
  SizeMember c[4]   = {};
  SizeFunction d[4] = {};

  assert(cuda::std::ranges::size(a) == 4);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::size(a)), size_t>);
  assert(cuda::std::ranges::size(b) == 1);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::size(b)), size_t>);
  assert(cuda::std::ranges::size(c) == 4);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::size(c)), size_t>);
  assert(cuda::std::ranges::size(d) == 4);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::size(d)), size_t>);

  return true;
}

struct SizeMemberConst
{
  __host__ __device__ constexpr size_t size() const
  {
    return 42;
  }
};

struct SizeMemberSigned
{
  __host__ __device__ constexpr long size()
  {
    return 42;
  }
};

__host__ __device__ bool constexpr testHasSizeMember()
{
  assert(cuda::std::ranges::size(SizeMember()) == 42);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::size(SizeMember())), size_t>);

  const SizeMemberConst sizeMemberConst{};
  assert(cuda::std::ranges::size(sizeMemberConst) == 42);

  assert(cuda::std::ranges::size(SizeMemberAndFunction()) == 42);

  assert(cuda::std::ranges::size(SizeMemberSigned()) == 42);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::size(SizeMemberSigned())), long>);

  assert(cuda::std::ranges::size(StaticSizeMember()) == 42);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::size(StaticSizeMember())), size_t>);

  return true;
}

struct MoveOnlySizeFunction
{
  MoveOnlySizeFunction()                            = default;
  MoveOnlySizeFunction(MoveOnlySizeFunction&&)      = default;
  MoveOnlySizeFunction(MoveOnlySizeFunction const&) = delete;

  __host__ __device__ friend constexpr size_t size(MoveOnlySizeFunction)
  {
    return 42;
  }
};

enum EnumSizeFunction
{
  a,
  b
};

__host__ __device__ constexpr size_t size(EnumSizeFunction)
{
  return 42;
}

struct SizeFunctionConst
{
  __host__ __device__ friend constexpr size_t size(const SizeFunctionConst)
  {
    return 42;
  }
};

struct SizeFunctionRef
{
  __host__ __device__ friend constexpr size_t size(SizeFunctionRef&)
  {
    return 42;
  }
};

struct SizeFunctionConstRef
{
  __host__ __device__ friend constexpr size_t size(SizeFunctionConstRef const&)
  {
    return 42;
  }
};

struct SizeFunctionSigned
{
  __host__ __device__ friend constexpr long size(SizeFunctionSigned)
  {
    return 42;
  }
};

__host__ __device__ bool constexpr testHasSizeFunction()
{
  assert(cuda::std::ranges::size(SizeFunction()) == 42);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::size(SizeFunction())), size_t>);
  static_assert(!cuda::std::is_invocable_v<RangeSizeT, MoveOnlySizeFunction>, "");
  assert(cuda::std::ranges::size(EnumSizeFunction()) == 42);
  assert(cuda::std::ranges::size(SizeFunctionConst()) == 42);

  SizeFunctionRef a{};
  assert(cuda::std::ranges::size(a) == 42);

  const SizeFunctionConstRef b{};
  assert(cuda::std::ranges::size(b) == 42);

  assert(cuda::std::ranges::size(SizeFunctionSigned()) == 42);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::size(SizeFunctionSigned())), long>);

  return true;
}

struct Empty
{};
static_assert(!cuda::std::is_invocable_v<RangeSizeT, Empty>, "");

struct InvalidReturnTypeMember
{
  __host__ __device__ Empty size();
};

struct InvalidReturnTypeFunction
{
  __host__ __device__ friend Empty size(InvalidReturnTypeFunction);
};

struct Convertible
{
  __host__ __device__ operator size_t();
};

struct ConvertibleReturnTypeMember
{
  __host__ __device__ Convertible size();
};

struct ConvertibleReturnTypeFunction
{
  __host__ __device__ friend Convertible size(ConvertibleReturnTypeFunction);
};

struct BoolReturnTypeMember
{
  __host__ __device__ bool size() const;
};

struct BoolReturnTypeFunction
{
  __host__ __device__ friend bool size(BoolReturnTypeFunction const&);
};

static_assert(!cuda::std::is_invocable_v<RangeSizeT, InvalidReturnTypeMember>, "");
static_assert(!cuda::std::is_invocable_v<RangeSizeT, InvalidReturnTypeFunction>, "");
static_assert(cuda::std::is_invocable_v<RangeSizeT, InvalidReturnTypeMember (&)[4]>, "");
static_assert(cuda::std::is_invocable_v<RangeSizeT, InvalidReturnTypeFunction (&)[4]>, "");
static_assert(!cuda::std::is_invocable_v<RangeSizeT, ConvertibleReturnTypeMember>, "");
static_assert(!cuda::std::is_invocable_v<RangeSizeT, ConvertibleReturnTypeFunction>, "");
static_assert(!cuda::std::is_invocable_v<RangeSizeT, BoolReturnTypeMember const&>, "");
static_assert(!cuda::std::is_invocable_v<RangeSizeT, BoolReturnTypeFunction const&>, "");

struct SizeMemberDisabled
{
  __host__ __device__ size_t size()
  {
    return 42;
  }
};

namespace cuda::std::ranges
{
template <>
inline constexpr bool disable_sized_range<SizeMemberDisabled> = true;
}

struct ImproperlyDisabledMember
{
  __host__ __device__ size_t size() const
  {
    return 42;
  }
};

// Intentionally disabling "const ConstSizeMemberDisabled". This doesn't disable anything
// because T is always uncvrefed before being checked.

namespace cuda::std::ranges
{
template <>
inline constexpr bool disable_sized_range<const ImproperlyDisabledMember> = true;
}

struct SizeFunctionDisabled
{
  __host__ __device__ friend size_t size(SizeFunctionDisabled)
  {
    return 42;
  }
};

namespace cuda::std::ranges
{
template <>
inline constexpr bool disable_sized_range<SizeFunctionDisabled> = true;
}

struct ImproperlyDisabledFunction
{
  __host__ __device__ friend size_t size(ImproperlyDisabledFunction const&)
  {
    return 42;
  }
};

namespace cuda::std::ranges
{
template <>
inline constexpr bool disable_sized_range<const ImproperlyDisabledFunction> = true;
}

static_assert(cuda::std::is_invocable_v<RangeSizeT, ImproperlyDisabledMember&>, "");
static_assert(cuda::std::is_invocable_v<RangeSizeT, const ImproperlyDisabledMember&>, "");
static_assert(!cuda::std::is_invocable_v<RangeSizeT, ImproperlyDisabledFunction&>, "");
static_assert(cuda::std::is_invocable_v<RangeSizeT, const ImproperlyDisabledFunction&>, "");

// No begin end.
struct HasMinusOperator
{
  __host__ __device__ friend constexpr size_t operator-(HasMinusOperator, HasMinusOperator)
  {
    return 2;
  }
};
static_assert(!cuda::std::is_invocable_v<RangeSizeT, HasMinusOperator>, "");

struct HasMinusBeginEnd
{
  struct sentinel
  {
    __host__ __device__ friend bool operator==(sentinel, forward_iterator<int*>);
#if TEST_STD_VER < 2020
    __host__ __device__ friend bool operator==(forward_iterator<int*>, sentinel);
    __host__ __device__ friend bool operator!=(sentinel, forward_iterator<int*>);
    __host__ __device__ friend bool operator!=(forward_iterator<int*>, sentinel);
#endif
    __host__ __device__ friend constexpr cuda::std::ptrdiff_t operator-(const sentinel, const forward_iterator<int*>)
    {
      return 2;
    }
    __host__ __device__ friend constexpr cuda::std::ptrdiff_t operator-(const forward_iterator<int*>, const sentinel)
    {
      return 2;
    }
  };

  __host__ __device__ friend constexpr forward_iterator<int*> begin(HasMinusBeginEnd)
  {
    return {};
  }
  __host__ __device__ friend constexpr sentinel end(HasMinusBeginEnd)
  {
    return {};
  }
};

struct other_forward_iterator : forward_iterator<int*>
{};

struct InvalidMinusBeginEnd
{
  struct sentinel
  {
    __host__ __device__ friend bool operator==(sentinel, other_forward_iterator);
#if TEST_STD_VER < 2020
    __host__ __device__ friend bool operator==(other_forward_iterator, sentinel);
    __host__ __device__ friend bool operator!=(sentinel, other_forward_iterator);
    __host__ __device__ friend bool operator!=(other_forward_iterator, sentinel);
#endif
    __host__ __device__ friend constexpr cuda::std::ptrdiff_t operator-(const sentinel, const other_forward_iterator)
    {
      return 2;
    }
    __host__ __device__ friend constexpr cuda::std::ptrdiff_t operator-(const other_forward_iterator, const sentinel)
    {
      return 2;
    }
  };

  __host__ __device__ friend constexpr other_forward_iterator begin(InvalidMinusBeginEnd)
  {
    return {};
  }
  __host__ __device__ friend constexpr sentinel end(InvalidMinusBeginEnd)
  {
    return {};
  }
};

// short is integer-like, but it is not other_forward_iterator's difference_type.
static_assert(!cuda::std::same_as<other_forward_iterator::difference_type, short>, "");
static_assert(!cuda::std::is_invocable_v<RangeSizeT, InvalidMinusBeginEnd>, "");

struct RandomAccessRange
{
  struct sentinel
  {
    __host__ __device__ friend bool operator==(sentinel, random_access_iterator<int*>);
#if TEST_STD_VER < 2020
    __host__ __device__ friend bool operator==(random_access_iterator<int*>, sentinel);
    __host__ __device__ friend bool operator!=(sentinel, random_access_iterator<int*>);
    __host__ __device__ friend bool operator!=(random_access_iterator<int*>, sentinel);
#endif
    __host__ __device__ friend constexpr cuda::std::ptrdiff_t
    operator-(const sentinel, const random_access_iterator<int*>)
    {
      return 2;
    }
    __host__ __device__ friend constexpr cuda::std::ptrdiff_t
    operator-(const random_access_iterator<int*>, const sentinel)
    {
      return 2;
    }
  };

  __host__ __device__ constexpr random_access_iterator<int*> begin()
  {
    return {};
  }
  __host__ __device__ constexpr sentinel end()
  {
    return {};
  }
};

struct IntPtrBeginAndEnd
{
  int buff[8];
  __host__ __device__ constexpr int* begin()
  {
    return buff;
  }
  __host__ __device__ constexpr int* end()
  {
    return buff + 8;
  }
};

struct DisabledSizeRangeWithBeginEnd
{
  int buff[8];
  __host__ __device__ constexpr int* begin()
  {
    return buff;
  }
  __host__ __device__ constexpr int* end()
  {
    return buff + 8;
  }
  __host__ __device__ constexpr size_t size()
  {
    return 1;
  }
};

namespace cuda::std::ranges
{
template <>
inline constexpr bool disable_sized_range<DisabledSizeRangeWithBeginEnd> = true;
}

struct SizeBeginAndEndMembers
{
  int buff[8];
  __host__ __device__ constexpr int* begin()
  {
    return buff;
  }
  __host__ __device__ constexpr int* end()
  {
    return buff + 8;
  }
  __host__ __device__ constexpr size_t size()
  {
    return 1;
  }
};

__host__ __device__ constexpr bool testRanges()
{
  HasMinusBeginEnd a{};
  assert(cuda::std::ranges::size(a) == 2);
  // Ensure that this is converted to an *unsigned* type.
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::size(a)), size_t>);

  IntPtrBeginAndEnd b{};
  assert(cuda::std::ranges::size(b) == 8);

  DisabledSizeRangeWithBeginEnd c{};
  assert(cuda::std::ranges::size(c) == 8);

  RandomAccessRange d{};
  assert(cuda::std::ranges::size(d) == 2);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::size(d)), size_t>);

  SizeBeginAndEndMembers e{};
  assert(cuda::std::ranges::size(e) == 1);

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
static_assert(!cuda::std::is_invocable_v<RangeSizeT, Holder<Incomplete>*>, "");
static_assert(!cuda::std::is_invocable_v<RangeSizeT, Holder<Incomplete>*&>, "");
#endif // TEST_STD_VER > 2017

int main(int, char**)
{
  testArrayType();
  static_assert(testArrayType(), "");

  testHasSizeMember();
  static_assert(testHasSizeMember(), "");

  testHasSizeFunction();
  static_assert(testHasSizeFunction(), "");

  testRanges();
  static_assert(testRanges(), "");

  return 0;
}
