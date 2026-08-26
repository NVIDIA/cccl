//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Utilities for source_location
 */

#pragma once

#include <cuda/__cccl_config>
#include <cuda/std/type_traits>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/source_location>

#include <cuda/experimental/__stf/utility/unittest.cuh>

namespace cuda::experimental::stf
{
/**
 * @brief Value (or reference) plus the call-site `source_location` of its construction.
 *
 * Intended as a function parameter type: write `with_location<widget>` instead of `widget`
 * so the callee can report file/line. Construction from an argument captures
 * `source_location::current()` at that call site (overloaded operators cannot take
 * defaulted `source_location` parameters themselves).
 *
 * Move-only: may be constructed as a temporary and moved into a by-value parameter,
 * but not copied. `T` may be a value, lvalue reference, or rvalue reference.
 */
template <class T>
struct with_location
{
  with_location(const with_location&) = delete;

  // Required so a converting temporary can initialize a by-value parameter.
  with_location(with_location&&) = default;

  // Constrained so that ill-formed reference bindings are detected by
  // `is_constructible_v` instead of erroring inside the mem-initializer, and so
  // that this template does not hijack the move constructor.
  template <typename U,
            ::cuda::std::enable_if_t<!::cuda::std::is_same_v<::cuda::std::decay_t<U>, with_location>
                                       && ::cuda::std::is_constructible_v<T, U&&>,
                                     int> = 0>
  constexpr with_location(U&& payload, ::cuda::std::source_location loc = ::cuda::std::source_location::current())
      : payload(::cuda::std::forward<U>(payload))
      , loc(loc)
  {}

  T payload;
  const ::cuda::std::source_location loc;
};

// Two-arg form only: CTAD cannot see `T` in the converting constructor, and a
// one-arg guide would steal moves (`with_location{std::move(w)}`).
template <class U>
with_location(U&&, ::cuda::std::source_location) -> with_location<::cuda::std::decay_t<U>>;

namespace reserved
{
struct source_location_hash
{
  /* We use const char * and not string because these are string literals,
   * and it is safe to assume they are not going to change. We also take the
   * function name into account because the same callsite could be used in
   * different instantiation of the same templated class, the name will reflect
   * the template parameters. */
  ::std::size_t operator()(const ::cuda::std::source_location& loc) const noexcept
  {
    return ::std::hash<const char*>{}(loc.file_name()) ^ (::std::hash<uint_least32_t>{}(loc.line()) << 1)
         ^ (::std::hash<uint_least32_t>{}(loc.column()) << 2) ^ (::std::hash<const char*>{}(loc.function_name()) << 3);
  }
};

struct source_location_equal
{
  bool operator()(const ::cuda::std::source_location& lhs, const ::cuda::std::source_location& rhs) const noexcept
  {
    // Comparing const char * is legit here because these are string literal constants
    return lhs.file_name() == rhs.file_name() && lhs.line() == rhs.line() && lhs.column() == rhs.column()
        && lhs.function_name() == rhs.function_name();
  }
};
} // namespace reserved
} // namespace cuda::experimental::stf

#ifdef UNITTESTED_FILE
UNITTEST("with_location")
{
  using cuda::experimental::stf::with_location;

  struct widget
  {
    int x = 0;
  };

  // Move-only wrapper (independent of whether T is copyable).
  static_assert(!::cuda::std::is_copy_constructible_v<with_location<widget>>);
  static_assert(!::cuda::std::is_copy_assignable_v<with_location<widget>>);
  static_assert(::cuda::std::is_move_constructible_v<with_location<widget>>);
  static_assert(!::cuda::std::is_move_assignable_v<with_location<widget>>);
  static_assert(!::cuda::std::is_default_constructible_v<with_location<widget>>);

  // Value T: takes lvalues (copy) or rvalues (move).
  static_assert(::cuda::std::is_constructible_v<with_location<widget>, widget>);
  static_assert(::cuda::std::is_constructible_v<with_location<widget>, widget&>);
  static_assert(::cuda::std::is_constructible_v<with_location<widget>, const widget&>);
  static_assert(::cuda::std::is_constructible_v<with_location<widget>, widget&&>);

  // Lvalue-reference T: binds only to lvalues.
  static_assert(::cuda::std::is_constructible_v<with_location<widget&>, widget&>);
  static_assert(!::cuda::std::is_constructible_v<with_location<widget&>, widget>);
  static_assert(!::cuda::std::is_constructible_v<with_location<widget&>, widget&&>);
  static_assert(::cuda::std::is_move_constructible_v<with_location<widget&>>);

  // Rvalue-reference T: binds only to rvalues.
  static_assert(::cuda::std::is_constructible_v<with_location<widget&&>, widget>);
  static_assert(::cuda::std::is_constructible_v<with_location<widget&&>, widget&&>);
  static_assert(!::cuda::std::is_constructible_v<with_location<widget&&>, widget&>);
  static_assert(!::cuda::std::is_constructible_v<with_location<widget&&>, const widget&>);
  static_assert(::cuda::std::is_move_constructible_v<with_location<widget&&>>);

  {
    const auto loc = ::cuda::std::source_location::current();
    auto wl        = with_location{widget{1}, loc};
    static_assert(::cuda::std::is_same_v<decltype(wl), with_location<widget>>);
    EXPECT(wl.loc.line() == loc.line());
  }
  auto consume_value = [](with_location<widget> w) {
    EXPECT(w.payload.x == 42);
    EXPECT(w.loc.line() != 0);
  };
  consume_value(widget{42});

  widget live{7};
  auto consume_lref = [](with_location<widget&> w) {
    EXPECT(w.payload.x == 7);
    w.payload.x = 9;
  };
  consume_lref(live);
  EXPECT(live.x == 9);

  auto consume_rref = [](with_location<widget&&> w) {
    EXPECT(w.payload.x == 3);
  };
  consume_rref(widget{3});
};
#endif // UNITTESTED_FILE
