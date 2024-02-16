//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef LIBCUDACXX_TEST_STD_ITERATOR_UNQUALIFIED_LOOKUP_WRAPPER
#define LIBCUDACXX_TEST_STD_ITERATOR_UNQUALIFIED_LOOKUP_WRAPPER

#include <cuda/std/iterator>
#include <cuda/std/utility>

#include "test_macros.h"

namespace check_unqualified_lookup {
// Wrapper around an iterator for testing unqualified calls to `iter_move` and `iter_swap`.
template <typename I>
class unqualified_lookup_wrapper {
public:
  unqualified_lookup_wrapper() = default;

  TEST_HOST_DEVICE constexpr explicit unqualified_lookup_wrapper(I i) noexcept : base_(cuda::std::move(i)) {}

  TEST_HOST_DEVICE constexpr decltype(auto) operator*() const noexcept { return *base_; }

  TEST_HOST_DEVICE constexpr unqualified_lookup_wrapper& operator++() noexcept {
    ++base_;
    return *this;
  }

  TEST_HOST_DEVICE constexpr void operator++(int) noexcept { ++base_; }

  TEST_HOST_DEVICE constexpr bool operator==(unqualified_lookup_wrapper const& other) const noexcept {
    return base_ == other.base_;
  }
#if TEST_STD_VER < 2020
  TEST_HOST_DEVICE constexpr bool operator!=(unqualified_lookup_wrapper const& other) const noexcept {
    return base_ != other.base_;
  }
#endif

  // Delegates `cuda::std::ranges::iter_move` for the underlying iterator. `noexcept(false)` will be used
  // to ensure that the unqualified-lookup overload is chosen.
  TEST_HOST_DEVICE friend constexpr decltype(auto) iter_move(unqualified_lookup_wrapper& i) noexcept(false) {
    return cuda::std::ranges::iter_move(i.base_);
  }

private:
  I base_ = I{};
};

enum unscoped_enum { a, b, c };
TEST_HOST_DEVICE constexpr unscoped_enum iter_move(unscoped_enum& e) noexcept(false) { return e; }

enum class scoped_enum { a, b, c };
TEST_HOST_DEVICE constexpr scoped_enum* iter_move(scoped_enum&) noexcept { return nullptr; }

union some_union {
  int x;
  double y;
};
TEST_HOST_DEVICE constexpr int iter_move(some_union& u) noexcept(false) { return u.x; }

} // namespace check_unqualified_lookup

class move_tracker {
public:
  move_tracker() = default;
  TEST_HOST_DEVICE constexpr move_tracker(move_tracker&& other) noexcept : moves_{other.moves_ + 1} { other.moves_ = 0; }
  TEST_HOST_DEVICE constexpr move_tracker& operator=(move_tracker&& other) noexcept {
    moves_ = other.moves_ + 1;
    other.moves_ = 0;
    return *this;
  }

  move_tracker(move_tracker const& other) = delete;
  move_tracker& operator=(move_tracker const& other) = delete;

  TEST_HOST_DEVICE constexpr int moves() const noexcept { return moves_; }

private:
  int moves_ = 0;
};

#endif // LIBCUDACXX_TEST_STD_ITERATOR_UNQUALIFIED_LOOKUP_WRAPPER
