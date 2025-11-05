//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16

// <ranges>

// template<class T>
// inline constexpr bool enable_view = ...;

#include <cuda/std/ranges>

#include "test_macros.h"

// Doesn't derive from view_base
struct Empty
{};
static_assert(!cuda::std::ranges::enable_view<Empty>, "");
static_assert(!cuda::std::ranges::enable_view<Empty&>, "");
static_assert(!cuda::std::ranges::enable_view<Empty&&>, "");
static_assert(!cuda::std::ranges::enable_view<const Empty>, "");
static_assert(!cuda::std::ranges::enable_view<const Empty&>, "");
static_assert(!cuda::std::ranges::enable_view<const Empty&&>, "");

// Derives from view_base, but privately
struct PrivateViewBase : private cuda::std::ranges::view_base
{};
static_assert(!cuda::std::ranges::enable_view<PrivateViewBase>, "");
static_assert(!cuda::std::ranges::enable_view<PrivateViewBase&>, "");
static_assert(!cuda::std::ranges::enable_view<PrivateViewBase&&>, "");
static_assert(!cuda::std::ranges::enable_view<const PrivateViewBase>, "");
static_assert(!cuda::std::ranges::enable_view<const PrivateViewBase&>, "");
static_assert(!cuda::std::ranges::enable_view<const PrivateViewBase&&>, "");

// Derives from view_base, but specializes enable_view to false
struct EnableViewFalse : cuda::std::ranges::view_base
{};

namespace cuda::std::ranges
{
template <>
constexpr bool enable_view<EnableViewFalse> = false;
} // namespace cuda::std::ranges

static_assert(!cuda::std::ranges::enable_view<EnableViewFalse>, "");
static_assert(!cuda::std::ranges::enable_view<EnableViewFalse&>, "");
static_assert(!cuda::std::ranges::enable_view<EnableViewFalse&&>, "");
static_assert(cuda::std::ranges::enable_view<const EnableViewFalse>, "");
static_assert(!cuda::std::ranges::enable_view<const EnableViewFalse&>, "");
static_assert(!cuda::std::ranges::enable_view<const EnableViewFalse&&>, "");

// Derives from view_base
struct PublicViewBase : cuda::std::ranges::view_base
{};
static_assert(cuda::std::ranges::enable_view<PublicViewBase>, "");
static_assert(!cuda::std::ranges::enable_view<PublicViewBase&>, "");
static_assert(!cuda::std::ranges::enable_view<PublicViewBase&&>, "");
static_assert(cuda::std::ranges::enable_view<const PublicViewBase>, "");
static_assert(!cuda::std::ranges::enable_view<const PublicViewBase&>, "");
static_assert(!cuda::std::ranges::enable_view<const PublicViewBase&&>, "");

// Does not derive from view_base, but specializes enable_view to true
struct EnableViewTrue
{};

namespace cuda::std::ranges
{
template <>
constexpr bool enable_view<EnableViewTrue> = true;
}

static_assert(cuda::std::ranges::enable_view<EnableViewTrue>, "");
static_assert(!cuda::std::ranges::enable_view<EnableViewTrue&>, "");
static_assert(!cuda::std::ranges::enable_view<EnableViewTrue&&>, "");
static_assert(!cuda::std::ranges::enable_view<const EnableViewTrue>, "");
static_assert(!cuda::std::ranges::enable_view<const EnableViewTrue&>, "");
static_assert(!cuda::std::ranges::enable_view<const EnableViewTrue&&>, "");

// Make sure that enable_view is a bool, not some other contextually-convertible-to-bool type.
static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::enable_view<Empty>), const bool>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::enable_view<PublicViewBase>), const bool>);

// view_interface requires c++17

struct V1 : cuda::std::ranges::view_interface<V1>
{};
static_assert(cuda::std::ranges::enable_view<V1>, "");
static_assert(!cuda::std::ranges::enable_view<V1&>, "");
static_assert(!cuda::std::ranges::enable_view<V1&&>, "");
static_assert(cuda::std::ranges::enable_view<const V1>, "");
static_assert(!cuda::std::ranges::enable_view<const V1&>, "");
static_assert(!cuda::std::ranges::enable_view<const V1&&>, "");

struct V2
    : cuda::std::ranges::view_interface<V1>
    , cuda::std::ranges::view_interface<V2>
{};
#if !TEST_COMPILER(MSVC) || TEST_STD_VER > 2017 // MSVC seems to allow the conversion
                                                // despite the ambiguity in
// C++17
static_assert(!cuda::std::ranges::enable_view<V2>, "");
#endif // !TEST_COMPILER(MSVC) || TEST_STD_VER > 2017
static_assert(!cuda::std::ranges::enable_view<V2&>, "");
static_assert(!cuda::std::ranges::enable_view<V2&&>, "");
#if !TEST_COMPILER(MSVC) || TEST_STD_VER > 2017 // MSVC seems to allow the conversion
                                                // despite the ambiguity in
// C++17
static_assert(!cuda::std::ranges::enable_view<const V2>, "");
#endif // !TEST_COMPILER(MSVC) || TEST_STD_VER > 2017
static_assert(!cuda::std::ranges::enable_view<const V2&>, "");
static_assert(!cuda::std::ranges::enable_view<const V2&&>, "");

struct V3 : cuda::std::ranges::view_interface<V1>
{};
static_assert(cuda::std::ranges::enable_view<V3>, "");
static_assert(!cuda::std::ranges::enable_view<V3&>, "");
static_assert(!cuda::std::ranges::enable_view<V3&&>, "");
static_assert(cuda::std::ranges::enable_view<const V3>, "");
static_assert(!cuda::std::ranges::enable_view<const V3&>, "");
static_assert(!cuda::std::ranges::enable_view<const V3&&>, "");

struct PrivateInherit : private cuda::std::ranges::view_interface<PrivateInherit>
{};
static_assert(!cuda::std::ranges::enable_view<PrivateInherit>, "");

#if TEST_STD_VER > 2017
// ADL-proof
struct Incomplete;
template <class T>
struct Holder
{
  T t;
};
static_assert(!cuda::std::ranges::enable_view<Holder<Incomplete>*>, "");
#endif

static_assert(!cuda::std::ranges::enable_view<void>, "");

int main(int, char**)
{
  return 0;
}
