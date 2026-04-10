//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_CONTAINER_SEQUENCES_INPLACE_VECTOR_TYPES_H
#define TEST_CONTAINER_SEQUENCES_INPLACE_VECTOR_TYPES_H

#include <cuda/std/array>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

struct Trivial
{
  int val_;

  Trivial() = default;
  TEST_FUNC constexpr Trivial(const int val) noexcept
      : val_(val)
  {}

  TEST_FUNC friend constexpr bool operator==(const Trivial& lhs, const Trivial& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  TEST_FUNC friend constexpr bool operator<(const Trivial& lhs, const Trivial& rhs) noexcept
  {
    return lhs.val_ < rhs.val_;
  }
};

struct NonTrivial
{
  int val_;

  TEST_FUNC constexpr NonTrivial() noexcept
      : val_(0)
  {}
  TEST_FUNC constexpr NonTrivial(const int val) noexcept
      : val_(val)
  {}
  TEST_FUNC friend constexpr bool operator==(const NonTrivial& lhs, const NonTrivial& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  TEST_FUNC friend constexpr bool operator<(const NonTrivial& lhs, const NonTrivial& rhs) noexcept
  {
    return lhs.val_ < rhs.val_;
  }
};

struct NonTrivialDestructor
{
  int val_;

  TEST_FUNC NonTrivialDestructor() noexcept
      : val_(0)
  {}
  TEST_FUNC NonTrivialDestructor(const int val) noexcept
      : val_(val)
  {}
  NonTrivialDestructor(const NonTrivialDestructor&)            = default;
  NonTrivialDestructor(NonTrivialDestructor&&)                 = default;
  NonTrivialDestructor& operator=(const NonTrivialDestructor&) = default;
  NonTrivialDestructor& operator=(NonTrivialDestructor&&)      = default;
  TEST_FUNC ~NonTrivialDestructor() noexcept {}
  TEST_FUNC friend bool operator==(const NonTrivialDestructor& lhs, const NonTrivialDestructor& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  TEST_FUNC friend bool operator<(const NonTrivialDestructor& lhs, const NonTrivialDestructor& rhs) noexcept
  {
    return lhs.val_ < rhs.val_;
  }
};
static_assert(!cuda::std::is_trivially_copy_constructible<NonTrivialDestructor>::value);
static_assert(!cuda::std::is_trivially_move_constructible<NonTrivialDestructor>::value);
static_assert(cuda::std::is_trivially_copy_assignable<NonTrivialDestructor>::value);
static_assert(cuda::std::is_trivially_move_assignable<NonTrivialDestructor>::value);

struct ThrowingDefaultConstruct
{
  int val_;

  TEST_FUNC constexpr ThrowingDefaultConstruct() noexcept(false)
      : val_(0)
  {}
  TEST_FUNC constexpr ThrowingDefaultConstruct(const int val) noexcept
      : val_(val)
  {}
  TEST_FUNC friend constexpr bool
  operator==(const ThrowingDefaultConstruct& lhs, const ThrowingDefaultConstruct& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  TEST_FUNC friend constexpr bool
  operator<(const ThrowingDefaultConstruct& lhs, const ThrowingDefaultConstruct& rhs) noexcept
  {
    return lhs.val_ < rhs.val_;
  }
};
static_assert(cuda::std::is_trivially_copy_constructible<ThrowingDefaultConstruct>::value);
static_assert(cuda::std::is_trivially_move_constructible<ThrowingDefaultConstruct>::value);
static_assert(cuda::std::is_trivially_copy_assignable<ThrowingDefaultConstruct>::value);
static_assert(cuda::std::is_trivially_move_assignable<ThrowingDefaultConstruct>::value);
#if !TEST_COMPILER(GCC, <, 10)
static_assert(!cuda::std::is_nothrow_default_constructible<ThrowingDefaultConstruct>::value);
#endif // !TEST_COMPILER(GCC, <, 10)

struct ThrowingCopyConstructor
{
  int val_;

  TEST_FUNC ThrowingCopyConstructor() noexcept
      : val_(0)
  {}
  TEST_FUNC ThrowingCopyConstructor(const int val) noexcept
      : val_(val)
  {}

  TEST_FUNC ThrowingCopyConstructor(const ThrowingCopyConstructor& other) noexcept(false)
      : val_(other.val_)
  {}
  ThrowingCopyConstructor(ThrowingCopyConstructor&&)                 = default;
  ThrowingCopyConstructor& operator=(const ThrowingCopyConstructor&) = default;
  ThrowingCopyConstructor& operator=(ThrowingCopyConstructor&&)      = default;

  TEST_FUNC friend bool operator==(const ThrowingCopyConstructor& lhs, const ThrowingCopyConstructor& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  TEST_FUNC friend bool operator<(const ThrowingCopyConstructor& lhs, const ThrowingCopyConstructor& rhs) noexcept
  {
    return lhs.val_ < rhs.val_;
  }
};
static_assert(!cuda::std::is_trivially_copy_constructible<ThrowingCopyConstructor>::value);
static_assert(cuda::std::is_trivially_move_constructible<ThrowingCopyConstructor>::value);
static_assert(cuda::std::is_trivially_copy_assignable<ThrowingCopyConstructor>::value);
static_assert(cuda::std::is_trivially_move_assignable<ThrowingCopyConstructor>::value);
static_assert(!cuda::std::is_nothrow_copy_constructible<ThrowingCopyConstructor>::value);

struct ThrowingMoveConstructor
{
  int val_;

  TEST_FUNC ThrowingMoveConstructor() noexcept
      : val_(0)
  {}
  TEST_FUNC ThrowingMoveConstructor(const int val) noexcept
      : val_(val)
  {}

  TEST_FUNC ThrowingMoveConstructor(ThrowingMoveConstructor&& other) noexcept(false)
      : val_(other.val_)
  {}
  ThrowingMoveConstructor(const ThrowingMoveConstructor&)            = default;
  ThrowingMoveConstructor& operator=(const ThrowingMoveConstructor&) = default;
  ThrowingMoveConstructor& operator=(ThrowingMoveConstructor&&)      = default;

  TEST_FUNC friend bool operator==(const ThrowingMoveConstructor& lhs, const ThrowingMoveConstructor& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  TEST_FUNC friend bool operator<(const ThrowingMoveConstructor& lhs, const ThrowingMoveConstructor& rhs) noexcept
  {
    return lhs.val_ < rhs.val_;
  }
};
static_assert(cuda::std::is_trivially_copy_constructible<ThrowingMoveConstructor>::value);
static_assert(!cuda::std::is_trivially_move_constructible<ThrowingMoveConstructor>::value);
static_assert(cuda::std::is_trivially_copy_assignable<ThrowingMoveConstructor>::value);
static_assert(cuda::std::is_trivially_move_assignable<ThrowingMoveConstructor>::value);
static_assert(!cuda::std::is_nothrow_move_constructible<ThrowingMoveConstructor>::value);

struct ThrowingCopyAssignment
{
  int val_;

  TEST_FUNC ThrowingCopyAssignment() noexcept
      : val_(0)
  {}
  TEST_FUNC ThrowingCopyAssignment(const int val) noexcept
      : val_(val)
  {}

  ThrowingCopyAssignment(const ThrowingCopyAssignment&) = default;
  ThrowingCopyAssignment(ThrowingCopyAssignment&&)      = default;
  TEST_FUNC ThrowingCopyAssignment& operator=(const ThrowingCopyAssignment& other) noexcept(false)
  {
    val_ = other.val_;
    return *this;
  }
  ThrowingCopyAssignment& operator=(ThrowingCopyAssignment&&) = default;

  TEST_FUNC friend bool operator==(const ThrowingCopyAssignment& lhs, const ThrowingCopyAssignment& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  TEST_FUNC friend bool operator<(const ThrowingCopyAssignment& lhs, const ThrowingCopyAssignment& rhs) noexcept
  {
    return lhs.val_ < rhs.val_;
  }
};
static_assert(cuda::std::is_trivially_copy_constructible<ThrowingCopyAssignment>::value);
static_assert(cuda::std::is_trivially_move_constructible<ThrowingCopyAssignment>::value);
static_assert(!cuda::std::is_trivially_copy_assignable<ThrowingCopyAssignment>::value);
static_assert(cuda::std::is_trivially_move_assignable<ThrowingCopyAssignment>::value);
static_assert(!cuda::std::is_nothrow_copy_assignable<ThrowingCopyAssignment>::value);

struct ThrowingMoveAssignment
{
  int val_;

  TEST_FUNC ThrowingMoveAssignment() noexcept
      : val_(0)
  {}
  TEST_FUNC ThrowingMoveAssignment(const int val) noexcept
      : val_(val)
  {}

  ThrowingMoveAssignment(ThrowingMoveAssignment&&)                 = default;
  ThrowingMoveAssignment(const ThrowingMoveAssignment&)            = default;
  ThrowingMoveAssignment& operator=(const ThrowingMoveAssignment&) = default;
  TEST_FUNC ThrowingMoveAssignment& operator=(ThrowingMoveAssignment&& other) noexcept(false)
  {
    val_ = other.val_;
    return *this;
  }

  TEST_FUNC friend bool operator==(const ThrowingMoveAssignment& lhs, const ThrowingMoveAssignment& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  TEST_FUNC friend bool operator<(const ThrowingMoveAssignment& lhs, const ThrowingMoveAssignment& rhs) noexcept
  {
    return lhs.val_ < rhs.val_;
  }
};
static_assert(cuda::std::is_trivially_copy_constructible<ThrowingMoveAssignment>::value);
static_assert(cuda::std::is_trivially_move_constructible<ThrowingMoveAssignment>::value);
static_assert(cuda::std::is_trivially_copy_assignable<ThrowingMoveAssignment>::value);
static_assert(!cuda::std::is_trivially_move_assignable<ThrowingMoveAssignment>::value);
static_assert(!cuda::std::is_nothrow_move_assignable<ThrowingMoveAssignment>::value);

struct ThrowingSwap
{
  int val_;

  TEST_FUNC ThrowingSwap() noexcept
      : val_(0)
  {}
  TEST_FUNC ThrowingSwap(const int val) noexcept
      : val_(val)
  {}

  TEST_FUNC friend bool operator==(const ThrowingSwap& lhs, const ThrowingSwap& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }

  TEST_FUNC void swap(ThrowingSwap& other) noexcept(false)
  {
    cuda::std::swap(val_, other.val_);
  }
};
static_assert(!cuda::std::is_nothrow_swappable<ThrowingMoveConstructor>::value);

template <class T, size_t Capacity>
struct input_range
{
  cuda::std::array<T, Capacity> data;
  cpp17_input_iterator<T*> end_{data.data() + Capacity};

  TEST_FUNC constexpr cpp17_input_iterator<T*> begin() noexcept
  {
    return cpp17_input_iterator<T*>{data.begin()};
  }

  TEST_FUNC constexpr sentinel_wrapper<cpp17_input_iterator<T*>> end() noexcept
  {
    return sentinel_wrapper<cpp17_input_iterator<T*>>{end_};
  }
};
static_assert(cuda::std::ranges::input_range<input_range<int, 4>>);
static_assert(!cuda::std::ranges::forward_range<input_range<int, 4>>);
static_assert(!cuda::std::ranges::common_range<input_range<int, 4>>);
static_assert(!cuda::std::ranges::sized_range<input_range<int, 4>>);

template <class T, size_t Capacity>
struct uncommon_range
{
  cuda::std::array<T, Capacity> data;
  forward_iterator<T*> end_{data.data() + Capacity};

  TEST_FUNC constexpr forward_iterator<T*> begin() noexcept
  {
    return forward_iterator<T*>{data.begin()};
  }

  TEST_FUNC constexpr sentinel_wrapper<forward_iterator<T*>> end() noexcept
  {
    return sentinel_wrapper<forward_iterator<T*>>{end_};
  }
};
static_assert(cuda::std::ranges::forward_range<uncommon_range<int, 4>>);
static_assert(!cuda::std::ranges::common_range<uncommon_range<int, 4>>);
static_assert(!cuda::std::ranges::sized_range<uncommon_range<int, 4>>);

template <class T, size_t Capacity>
struct sized_uncommon_range
{
  cuda::std::array<T, Capacity> data;
  forward_iterator<T*> end_{data.data() + Capacity};

  TEST_FUNC constexpr forward_iterator<T*> begin() noexcept
  {
    return forward_iterator<T*>{data.begin()};
  }

  TEST_FUNC constexpr sized_sentinel<forward_iterator<T*>> end() noexcept
  {
    return sized_sentinel<forward_iterator<T*>>{end_};
  }
};
static_assert(cuda::std::ranges::forward_range<sized_uncommon_range<int, 4>>);
static_assert(!cuda::std::ranges::common_range<sized_uncommon_range<int, 4>>);
static_assert(cuda::std::ranges::sized_range<sized_uncommon_range<int, 4>>);

// Helper function to compare two ranges
template <class Range1, class Range2>
TEST_FUNC constexpr bool equal_range(const Range1& range1, const Range2& range2)
{
  return cuda::std::equal(range1.begin(), range1.end(), range2.begin(), range2.end());
}

#endif // TEST_CONTAINER_SEQUENCES_INPLACE_VECTOR_TYPES_H
