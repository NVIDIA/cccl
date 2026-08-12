//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class F, class I1, class I2>
// concept __indirectly_regular_binary_invocable;

#include <cuda/std/concepts>
#include <cuda/std/iterator>

#include "indirectly_readable.h"
#include "test_macros.h"

using It1 = IndirectlyReadable<struct Token>;
using It2 = IndirectlyReadable2<struct Token>;
using R1  = T1<struct ReturnToken>;
using R2  = T2<struct ReturnToken>;

template <class I1, class I2>
struct GoodInvocable
{
  TEST_FUNC R1 operator()(cuda::std::iter_value_t<I1>&, cuda::std::iter_value_t<I2>&) const;
  TEST_FUNC R1 operator()(cuda::std::iter_value_t<I1>&, cuda::std::iter_reference_t<I2>) const;
  TEST_FUNC R1 operator()(cuda::std::iter_reference_t<I1>, cuda::std::iter_value_t<I2>&) const;
  TEST_FUNC R2 operator()(cuda::std::iter_reference_t<I1>, cuda::std::iter_reference_t<I2>) const;
  TEST_FUNC R2 operator()(cuda::std::iter_common_reference_t<I1>, cuda::std::iter_common_reference_t<I2>) const;
};

// Should work when all constraints are satisfied
static_assert(cuda::std::__indirectly_regular_binary_invocable<GoodInvocable<It1, It2>, It1, It2>);

// Should fail when the iterator is not indirectly_readable
#if TEST_STD_VER > 2017
struct NotIndirectlyReadable
{};
static_assert(
  !cuda::std::
    __indirectly_regular_binary_invocable<GoodInvocable<NotIndirectlyReadable, It2>, NotIndirectlyReadable, It2>);
#endif

// Should fail when the invocable is not copy constructible
struct BadInvocable1
{
  BadInvocable1(BadInvocable1 const&) = delete;
  template <class T1, class T2>
  TEST_FUNC R1 operator()(T1 const&, T2 const&) const;
};
static_assert(!cuda::std::__indirectly_regular_binary_invocable<BadInvocable1, It1, It2>);

// Should fail when the invocable can't be called with (cuda::std::iter_value_t<I1>&, cuda::std::iter_value_t<I2>&)
struct BadInvocable2
{
  template <class T1, class T2>
  TEST_FUNC R1 operator()(T1 const&, T2 const&) const;
  R1 operator()(cuda::std::iter_value_t<It1>&, cuda::std::iter_value_t<It2>&) const = delete;
};
static_assert(!cuda::std::__indirectly_regular_binary_invocable<BadInvocable2, It1, It2>);

// Should fail when the invocable can't be called with (cuda::std::iter_value_t<I1>&, cuda::std::iter_reference_t<I2>)
struct BadInvocable3
{
  template <class T1, class T2>
  TEST_FUNC R1 operator()(T1 const&, T2 const&) const;
  R1 operator()(cuda::std::iter_value_t<It1>&, cuda::std::iter_reference_t<It2>) const = delete;
};
static_assert(!cuda::std::__indirectly_regular_binary_invocable<BadInvocable3, It1, It2>);

// Should fail when the invocable can't be called with (cuda::std::iter_reference_t<I1>, cuda::std::iter_value_t<I2>&)
struct BadInvocable4
{
  template <class T1, class T2>
  TEST_FUNC R1 operator()(T1 const&, T2 const&) const;
  R1 operator()(cuda::std::iter_reference_t<It1>, cuda::std::iter_value_t<It2>&) const = delete;
};
static_assert(!cuda::std::__indirectly_regular_binary_invocable<BadInvocable4, It1, It2>);

// Should fail when the invocable can't be called with (cuda::std::iter_reference_t<I1>,
// cuda::std::iter_reference_t<I2>)
struct BadInvocable5
{
  template <class T1, class T2>
  TEST_FUNC R1 operator()(T1 const&, T2 const&) const;
  R1 operator()(cuda::std::iter_reference_t<It1>, cuda::std::iter_reference_t<It2>) const = delete;
};
static_assert(!cuda::std::__indirectly_regular_binary_invocable<BadInvocable5, It1, It2>);

// Should fail when the invocable can't be called with (iter_common_reference_t)
struct BadInvocable6
{
  template <class T1, class T2>
  TEST_FUNC R1 operator()(T1 const&, T2 const&) const;
  R1 operator()(cuda::std::iter_common_reference_t<It1>, cuda::std::iter_common_reference_t<It2>) const = delete;
};
static_assert(!cuda::std::__indirectly_regular_binary_invocable<BadInvocable6, It1, It2>);

// Should fail when the invocable doesn't have a common reference between its return types
struct BadInvocable7
{
  struct Unrelated
  {};
  TEST_FUNC Unrelated operator()(cuda::std::iter_value_t<It1>&, cuda::std::iter_value_t<It2>&) const;
  TEST_FUNC R1 operator()(cuda::std::iter_value_t<It1>&, cuda::std::iter_reference_t<It2>) const;
  TEST_FUNC R1 operator()(cuda::std::iter_reference_t<It1>, cuda::std::iter_value_t<It2>&) const;
  TEST_FUNC R1 operator()(cuda::std::iter_reference_t<It1>, cuda::std::iter_reference_t<It2>) const;
  TEST_FUNC R1 operator()(cuda::std::iter_common_reference_t<It1>, cuda::std::iter_common_reference_t<It2>) const;
};
static_assert(!cuda::std::__indirectly_regular_binary_invocable<BadInvocable7, It1, It2>);

// Should fail when the invocable doesn't have a common reference between its return types
struct BadInvocable8
{
  struct Unrelated
  {};
  TEST_FUNC R1 operator()(cuda::std::iter_value_t<It1>&, cuda::std::iter_value_t<It2>&) const;
  TEST_FUNC Unrelated operator()(cuda::std::iter_value_t<It1>&, cuda::std::iter_reference_t<It2>) const;
  TEST_FUNC R1 operator()(cuda::std::iter_reference_t<It1>, cuda::std::iter_value_t<It2>&) const;
  TEST_FUNC R1 operator()(cuda::std::iter_reference_t<It1>, cuda::std::iter_reference_t<It2>) const;
  TEST_FUNC R1 operator()(cuda::std::iter_common_reference_t<It1>, cuda::std::iter_common_reference_t<It2>) const;
};
static_assert(!cuda::std::__indirectly_regular_binary_invocable<BadInvocable8, It1, It2>);

// Should fail when the invocable doesn't have a common reference between its return types
struct BadInvocable9
{
  struct Unrelated
  {};
  TEST_FUNC R1 operator()(cuda::std::iter_value_t<It1>&, cuda::std::iter_value_t<It2>&) const;
  TEST_FUNC R1 operator()(cuda::std::iter_value_t<It1>&, cuda::std::iter_reference_t<It2>) const;
  TEST_FUNC Unrelated operator()(cuda::std::iter_reference_t<It1>, cuda::std::iter_value_t<It2>&) const;
  TEST_FUNC R1 operator()(cuda::std::iter_reference_t<It1>, cuda::std::iter_reference_t<It2>) const;
  TEST_FUNC R1 operator()(cuda::std::iter_common_reference_t<It1>, cuda::std::iter_common_reference_t<It2>) const;
};
static_assert(!cuda::std::__indirectly_regular_binary_invocable<BadInvocable9, It1, It2>);

// Should fail when the invocable doesn't have a common reference between its return types
struct BadInvocable10
{
  struct Unrelated
  {};
  TEST_FUNC R1 operator()(cuda::std::iter_value_t<It1>&, cuda::std::iter_value_t<It2>&) const;
  TEST_FUNC R1 operator()(cuda::std::iter_value_t<It1>&, cuda::std::iter_reference_t<It2>) const;
  TEST_FUNC R1 operator()(cuda::std::iter_reference_t<It1>, cuda::std::iter_value_t<It2>&) const;
  TEST_FUNC Unrelated operator()(cuda::std::iter_reference_t<It1>, cuda::std::iter_reference_t<It2>) const;
  TEST_FUNC R1 operator()(cuda::std::iter_common_reference_t<It1>, cuda::std::iter_common_reference_t<It2>) const;
};
static_assert(!cuda::std::__indirectly_regular_binary_invocable<BadInvocable10, It1, It2>);

// Should fail when the invocable doesn't have a common reference between its return types
struct BadInvocable11
{
  struct Unrelated
  {};
  TEST_FUNC R1 operator()(cuda::std::iter_value_t<It1>&, cuda::std::iter_value_t<It2>&) const;
  TEST_FUNC R1 operator()(cuda::std::iter_value_t<It1>&, cuda::std::iter_reference_t<It2>) const;
  TEST_FUNC R1 operator()(cuda::std::iter_reference_t<It1>, cuda::std::iter_value_t<It2>&) const;
  TEST_FUNC R1 operator()(cuda::std::iter_reference_t<It1>, cuda::std::iter_reference_t<It2>) const;
  TEST_FUNC Unrelated operator()(cuda::std::iter_common_reference_t<It1>, cuda::std::iter_common_reference_t<It2>) const;
};
static_assert(!cuda::std::__indirectly_regular_binary_invocable<BadInvocable11, It1, It2>);

// Various tests with callables
struct S
{};
static_assert(cuda::std::__indirectly_regular_binary_invocable<int (*)(int, double), int*, double*>);
static_assert(cuda::std::__indirectly_regular_binary_invocable<int (&)(int, double), int*, double*>);
static_assert(cuda::std::__indirectly_regular_binary_invocable<void (*)(int, double), int*, double*>);

static_assert(!cuda::std::__indirectly_regular_binary_invocable<int(int, double), int*, double*>); // not move
                                                                                                   // constructible
static_assert(!cuda::std::__indirectly_regular_binary_invocable<int (*)(int*, int*, int*), int*, int*>);
static_assert(!cuda::std::__indirectly_regular_binary_invocable<int (&)(int*, int*, int*), int*, int*>);
static_assert(!cuda::std::__indirectly_regular_binary_invocable<int (*)(int*), int*, int*>);
static_assert(!cuda::std::__indirectly_regular_binary_invocable<int (&)(int*), int*, int*>);

int main(int, char**)
{
  return 0;
}
