//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// <ranges>

// template<class T>
// concept view = ...;

#include <cuda/std/ranges>

#include "test_macros.h"

// The type would be a view, but it's not moveable.
struct NotMoveable : cuda::std::ranges::view_base {
  NotMoveable() = default;
  NotMoveable(NotMoveable&&) = delete;
  NotMoveable& operator=(NotMoveable&&) = delete;
  TEST_HOST_DEVICE friend int* begin(NotMoveable&);
  TEST_HOST_DEVICE friend int* begin(NotMoveable const&);
  TEST_HOST_DEVICE friend int* end(NotMoveable&);
  TEST_HOST_DEVICE friend int* end(NotMoveable const&);
};
static_assert(cuda::std::ranges::range<NotMoveable>);
static_assert(!cuda::std::movable<NotMoveable>);
static_assert(cuda::std::default_initializable<NotMoveable>);
static_assert(cuda::std::ranges::enable_view<NotMoveable>);
static_assert(!cuda::std::ranges::view<NotMoveable>);

// The type would be a view, but it's not default initializable
struct NotDefaultInit : cuda::std::ranges::view_base {
  NotDefaultInit() = delete;
  TEST_HOST_DEVICE friend int* begin(NotDefaultInit&);
  TEST_HOST_DEVICE friend int* begin(NotDefaultInit const&);
  TEST_HOST_DEVICE friend int* end(NotDefaultInit&);
  TEST_HOST_DEVICE friend int* end(NotDefaultInit const&);
};
static_assert(cuda::std::ranges::range<NotDefaultInit>);
static_assert(cuda::std::movable<NotDefaultInit>);
static_assert(!cuda::std::default_initializable<NotDefaultInit>);
static_assert(cuda::std::ranges::enable_view<NotDefaultInit>);
static_assert(cuda::std::ranges::view<NotDefaultInit>);

// The type would be a view, but it doesn't enable it with enable_view
struct NotExplicitlyEnabled {
  NotExplicitlyEnabled() = default;
  NotExplicitlyEnabled(NotExplicitlyEnabled&&) = default;
  NotExplicitlyEnabled& operator=(NotExplicitlyEnabled&&) = default;
  TEST_HOST_DEVICE friend int* begin(NotExplicitlyEnabled&);
  TEST_HOST_DEVICE friend int* begin(NotExplicitlyEnabled const&);
  TEST_HOST_DEVICE friend int* end(NotExplicitlyEnabled&);
  TEST_HOST_DEVICE friend int* end(NotExplicitlyEnabled const&);
};
static_assert(cuda::std::ranges::range<NotExplicitlyEnabled>);
static_assert(cuda::std::movable<NotExplicitlyEnabled>);
static_assert(cuda::std::default_initializable<NotExplicitlyEnabled>);
static_assert(!cuda::std::ranges::enable_view<NotExplicitlyEnabled>);
static_assert(!cuda::std::ranges::view<NotExplicitlyEnabled>);

// The type has everything else, but it's not a range
struct NotARange : cuda::std::ranges::view_base {
  NotARange() = default;
  NotARange(NotARange&&) = default;
  NotARange& operator=(NotARange&&) = default;
};
static_assert(!cuda::std::ranges::range<NotARange>);
static_assert(cuda::std::movable<NotARange>);
static_assert(cuda::std::default_initializable<NotARange>);
static_assert(cuda::std::ranges::enable_view<NotARange>);
static_assert(!cuda::std::ranges::view<NotARange>);

// The type satisfies all requirements
struct View : cuda::std::ranges::view_base {
  View() = default;
  View(View&&) = default;
  View& operator=(View&&) = default;
  TEST_HOST_DEVICE friend int* begin(View&);
  TEST_HOST_DEVICE friend int* begin(View const&);
  TEST_HOST_DEVICE friend int* end(View&);
  TEST_HOST_DEVICE friend int* end(View const&);
};
static_assert(cuda::std::ranges::range<View>);
static_assert(cuda::std::movable<View>);
static_assert(cuda::std::default_initializable<View>);
static_assert(cuda::std::ranges::enable_view<View>);
static_assert(cuda::std::ranges::view<View>);

int main(int, char**) {
  return 0;
}
