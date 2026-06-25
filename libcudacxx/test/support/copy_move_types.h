//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_UTILITIES_TUPLE_CNSTR_TYPES_H
#define TEST_STD_UTILITIES_TUPLE_CNSTR_TYPES_H

#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

#if !TEST_COMPILER(NVRTC)
#  include "test_allocator.h"
#endif // !TEST_COMPILER(NVRTC)

// Types that can be used to test copy/move operations

struct MutableCopy
{
  int val{};
  bool alloc_constructed{false};

  constexpr MutableCopy() = default;
  TEST_FUNC constexpr MutableCopy(int _val)
      : val(_val)
  {}
  constexpr MutableCopy(MutableCopy&) = default;
#if TEST_STD_VER >= 2020 || !TEST_COMPILER(MSVC)
  TEST_FUNC constexpr MutableCopy(const MutableCopy&) = delete;
#endif // TEST_STD_VER >= 2020 || !TEST_COMPILER(MSVC)

#if !TEST_COMPILER(NVRTC)
  TEST_FUNC constexpr MutableCopy(cuda::std::allocator_arg_t, const test_allocator<int>&, MutableCopy& o)
      : val(o.val)
      , alloc_constructed(true)
  {}
#endif // !TEST_COMPILER(NVRTC)
};

#if !TEST_COMPILER(NVRTC)
template <>
struct cuda::std::uses_allocator<MutableCopy, test_allocator<int>> : cuda::std::true_type
{};
#endif // !TEST_COMPILER(NVRTC)

struct ConstCopy
{
  int val{};
  bool alloc_constructed{false};

  constexpr ConstCopy() = default;
  TEST_FUNC constexpr ConstCopy(int _val)
      : val(_val)
  {}
  constexpr ConstCopy(const ConstCopy&) = default;
#if TEST_STD_VER >= 2020 || !TEST_COMPILER(MSVC)
  TEST_FUNC constexpr ConstCopy(ConstCopy&) = delete;
#endif // TEST_STD_VER >= 2020 || !TEST_COMPILER(MSVC)

#if !TEST_COMPILER(NVRTC)
  TEST_FUNC constexpr ConstCopy(cuda::std::allocator_arg_t, const test_allocator<int>&, const ConstCopy& o)
      : val(o.val)
      , alloc_constructed(true)
  {}
#endif // !TEST_COMPILER(NVRTC)
};

#if !TEST_COMPILER(NVRTC)
template <>
struct cuda::std::uses_allocator<ConstCopy, test_allocator<int>> : cuda::std::true_type
{};
#endif // !TEST_COMPILER(NVRTC)

struct MutableMove
{
  int val{};
  bool alloc_constructed{false};

  constexpr MutableMove() = default;
  TEST_FUNC constexpr MutableMove(int _val)
      : val(_val)
  {}
  constexpr MutableMove(MutableMove&&) = default;
#if TEST_STD_VER >= 2020 || !TEST_COMPILER(MSVC)
  TEST_FUNC constexpr MutableMove(const MutableMove&&) = delete;
#endif // TEST_STD_VER >= 2020 || !TEST_COMPILER(MSVC)

#if !TEST_COMPILER(NVRTC)
  TEST_FUNC constexpr MutableMove(cuda::std::allocator_arg_t, const test_allocator<int>&, MutableMove&& o)
      : val(o.val)
      , alloc_constructed(true)
  {}
#endif // !TEST_COMPILER(NVRTC)
};

#if !TEST_COMPILER(NVRTC)
template <>
struct cuda::std::uses_allocator<MutableMove, test_allocator<int>> : cuda::std::true_type
{};
#endif // !TEST_COMPILER(NVRTC)

struct ConstMove
{
  int val{};
  bool alloc_constructed{false};

  constexpr ConstMove() = default;
  TEST_FUNC constexpr ConstMove(int _val)
      : val(_val)
  {}
  TEST_FUNC constexpr ConstMove(const ConstMove&& o) noexcept
      : val(o.val)
  {}
#if TEST_STD_VER >= 2020 || !TEST_COMPILER(MSVC)
  TEST_FUNC constexpr ConstMove(ConstMove&&) = delete;
#endif // TEST_STD_VER >= 2020 || !TEST_COMPILER(MSVC)

#if !TEST_COMPILER(NVRTC)
  TEST_FUNC constexpr ConstMove(cuda::std::allocator_arg_t, const test_allocator<int>&, const ConstMove&& o)
      : val(o.val)
      , alloc_constructed(true)
  {}
#endif // !TEST_COMPILER(NVRTC)
};

#if !TEST_COMPILER(NVRTC)
template <>
struct cuda::std::uses_allocator<ConstMove, test_allocator<int>> : cuda::std::true_type
{};
#endif // !TEST_COMPILER(NVRTC)

template <class T, bool NothrowConstructible = true>
struct ConvertibleFrom
{
  T v{};
  bool alloc_constructed{false};

  constexpr ConvertibleFrom() = default;

  template <class U = T, cuda::std::enable_if_t<cuda::std::is_constructible_v<U, U&>, int> = 0>
  TEST_FUNC constexpr ConvertibleFrom(T& _v) noexcept(NothrowConstructible)
      : v(_v)
  {}

  template <class U                                                                                              = T,
            cuda::std::enable_if_t<cuda::std::is_constructible_v<U, const U&> && !cuda::std::is_const_v<U>, int> = 0>
  TEST_FUNC constexpr ConvertibleFrom(const T& _v) noexcept(NothrowConstructible)
      : v(_v)
  {}

  template <class U = T, cuda::std::enable_if_t<cuda::std::is_constructible_v<U, U&&>, int> = 0>
  TEST_FUNC constexpr ConvertibleFrom(T&& _v) noexcept(NothrowConstructible)
      : v(cuda::std::move(_v))
  {}

  template <class U                                                                                               = T,
            cuda::std::enable_if_t<cuda::std::is_constructible_v<U, const U&&> && !cuda::std::is_const_v<U>, int> = 0>
  TEST_FUNC constexpr ConvertibleFrom(const T&& _v) noexcept(NothrowConstructible)
      : v(cuda::std::move(_v))
  {}

#if !TEST_COMPILER(NVRTC)
  template <class U, cuda::std::enable_if_t<cuda::std::is_constructible_v<ConvertibleFrom, U&&>, int> = 0>
  TEST_FUNC constexpr ConvertibleFrom(cuda::std::allocator_arg_t, const test_allocator<int>&, U&& _u) noexcept(
    NothrowConstructible)
      : ConvertibleFrom{cuda::std::forward<U>(_u)}
  {
    alloc_constructed = true;
  }
#endif // !TEST_COMPILER(NVRTC)
};

#if !TEST_COMPILER(NVRTC)
template <class T>
struct cuda::std::uses_allocator<ConvertibleFrom<T>, test_allocator<int>> : cuda::std::true_type
{};
#endif // !TEST_COMPILER(NVRTC)

template <class T, bool NothrowConstructible = true>
struct ExplicitConstructibleFrom
{
  T v{};
  bool alloc_constructed{false};

  constexpr explicit ExplicitConstructibleFrom() = default;

  template <class U = T, cuda::std::enable_if_t<cuda::std::is_constructible_v<U, U&>, int> = 0>
  TEST_FUNC constexpr explicit ExplicitConstructibleFrom(T& _v) noexcept(NothrowConstructible)
      : v(_v)
  {}

  template <class U                                                                                              = T,
            cuda::std::enable_if_t<cuda::std::is_constructible_v<U, const U&> && !cuda::std::is_const_v<U>, int> = 0>
  TEST_FUNC constexpr explicit ExplicitConstructibleFrom(const T& _v) noexcept(NothrowConstructible)
      : v(_v)
  {}

  template <class U = T, cuda::std::enable_if_t<cuda::std::is_constructible_v<U, U&&>, int> = 0>
  TEST_FUNC constexpr explicit ExplicitConstructibleFrom(T&& _v) noexcept(NothrowConstructible)
      : v(cuda::std::move(_v))
  {}

  template <class U                                                                                               = T,
            cuda::std::enable_if_t<cuda::std::is_constructible_v<U, const U&&> && !cuda::std::is_const_v<U>, int> = 0>
  TEST_FUNC constexpr explicit ExplicitConstructibleFrom(const T&& _v) noexcept(NothrowConstructible)
      : v(cuda::std::move(_v))
  {}

#if !TEST_COMPILER(NVRTC)
  template <class U, cuda::std::enable_if_t<cuda::std::is_constructible_v<ExplicitConstructibleFrom, U&&>, int> = 0>
  TEST_FUNC constexpr ExplicitConstructibleFrom(
    cuda::std::allocator_arg_t, const test_allocator<int>&, U&& _u) noexcept(NothrowConstructible)
      : ExplicitConstructibleFrom{cuda::std::forward<U>(_u)}
  {
    alloc_constructed = true;
  }
#endif // !TEST_COMPILER(NVRTC)
};

#if !TEST_COMPILER(NVRTC)
template <class T>
struct cuda::std::uses_allocator<ExplicitConstructibleFrom<T>, test_allocator<int>> : cuda::std::true_type
{};
#endif // !TEST_COMPILER(NVRTC)

struct TracedCopyMove
{
  int nonConstCopy       = 0;
  int constCopy          = 0;
  int nonConstMove       = 0;
  int constMove          = 0;
  bool alloc_constructed = false;

  constexpr TracedCopyMove() = default;
  TEST_FUNC constexpr TracedCopyMove(const TracedCopyMove& other)
      : nonConstCopy(other.nonConstCopy)
      , constCopy(other.constCopy + 1)
      , nonConstMove(other.nonConstMove)
      , constMove(other.constMove)
  {}
  TEST_FUNC constexpr TracedCopyMove(TracedCopyMove& other)
      : nonConstCopy(other.nonConstCopy + 1)
      , constCopy(other.constCopy)
      , nonConstMove(other.nonConstMove)
      , constMove(other.constMove)
  {}

  TEST_FUNC constexpr TracedCopyMove(TracedCopyMove&& other) noexcept
      : nonConstCopy(other.nonConstCopy)
      , constCopy(other.constCopy)
      , nonConstMove(other.nonConstMove + 1)
      , constMove(other.constMove)
  {}

  TEST_FUNC constexpr TracedCopyMove(const TracedCopyMove&& other) noexcept
      : nonConstCopy(other.nonConstCopy)
      , constCopy(other.constCopy)
      , nonConstMove(other.nonConstMove)
      , constMove(other.constMove + 1)
  {}

#if !TEST_COMPILER(NVRTC)
  template <class U, cuda::std::enable_if_t<cuda::std::is_constructible_v<TracedCopyMove, U&&>, int> = 0>
  TEST_FUNC constexpr TracedCopyMove(cuda::std::allocator_arg_t, const test_allocator<int>&, U&& _u)
      : TracedCopyMove{cuda::std::forward<U>(_u)}
  {
    alloc_constructed = true;
  }
#endif // !TEST_COMPILER(NVRTC)
};

#if !TEST_COMPILER(NVRTC)
template <>
struct cuda::std::uses_allocator<TracedCopyMove, test_allocator<int>> : cuda::std::true_type
{};
#endif // !TEST_COMPILER(NVRTC)

// If the constructor tuple(tuple<UTypes...>&) is not available,
// the fallback call to `tuple(const tuple&) = default;` or any other
// constructor that takes const ref would increment the constCopy.
TEST_FUNC constexpr bool nonConstCopyCtrCalled(const TracedCopyMove& obj)
{
  return obj.nonConstCopy == 1 && obj.constCopy == 0 && obj.constMove == 0 && obj.nonConstMove == 0;
}

TEST_FUNC constexpr bool constCopyCtrCalled(const TracedCopyMove& obj)
{
  return obj.nonConstCopy == 0 && obj.constCopy == 1 && obj.constMove == 0 && obj.nonConstMove == 0;
}

TEST_FUNC constexpr bool moveCtrCalled(const TracedCopyMove& obj)
{
  return obj.nonConstMove == 1 && obj.constMove == 0 && obj.constCopy == 0 && obj.nonConstCopy == 0;
}

// If the constructor tuple(const tuple<UTypes...>&&) is not available,
// the fallback call to `tuple(const tuple&) = default;` or any other
// constructor that takes const ref would increment the constCopy.
TEST_FUNC constexpr bool constMoveCtrCalled(const TracedCopyMove& obj)
{
  return obj.nonConstMove == 0 && obj.constMove == 1 && obj.constCopy == 0 && obj.nonConstCopy == 0;
}

struct NoConstructorFromInt
{};

struct CvtFromTupleRef : TracedCopyMove
{
  constexpr CvtFromTupleRef() = default;
  TEST_FUNC constexpr CvtFromTupleRef(cuda::std::tuple<CvtFromTupleRef>& other)
      : TracedCopyMove(static_cast<TracedCopyMove&>(cuda::std::get<0>(other)))
  {}
};

struct ExplicitCtrFromTupleRef : TracedCopyMove
{
  constexpr explicit ExplicitCtrFromTupleRef() = default;
  TEST_FUNC constexpr explicit ExplicitCtrFromTupleRef(cuda::std::tuple<ExplicitCtrFromTupleRef>& other)
      : TracedCopyMove(static_cast<TracedCopyMove&>(cuda::std::get<0>(other)))
  {}
};

struct CvtFromConstTupleRefRef : TracedCopyMove
{
  constexpr CvtFromConstTupleRefRef() = default;
  TEST_FUNC constexpr CvtFromConstTupleRefRef(cuda::std::tuple<const CvtFromConstTupleRefRef>&& other)
      : TracedCopyMove(static_cast<const TracedCopyMove&&>(cuda::std::get<0>(other)))
  {}
};

struct ExplicitCtrFromConstTupleRefRef : TracedCopyMove
{
  constexpr explicit ExplicitCtrFromConstTupleRefRef() = default;
  TEST_FUNC constexpr explicit ExplicitCtrFromConstTupleRefRef(
    cuda::std::tuple<const ExplicitCtrFromConstTupleRefRef>&& other)
      : TracedCopyMove(static_cast<const TracedCopyMove&&>(cuda::std::get<0>(other)))
  {}
};

template <class T>
TEST_FUNC constexpr void conversion_test(T)
{}

template <class T, class... Args>
struct ImplicitlyConstructibleImpl
{
  template <class U, class... As>
  TEST_FUNC static constexpr auto test(int)
    -> decltype(conversion_test<U>({cuda::std::declval<As>()...}), cuda::std::true_type{});

  template <class, class...>
  TEST_FUNC static constexpr auto test(...) -> cuda::std::false_type;

  static constexpr bool value = decltype(test<T, Args...>(0))::value;
};

template <class T, class... Args>
inline constexpr bool ImplicitlyConstructible = ImplicitlyConstructibleImpl<T, Args...>::value;

struct CopyAssign
{
  int val{};

  constexpr CopyAssign() = default;
  TEST_FUNC constexpr CopyAssign(int v)
      : val(v)
  {}

  constexpr CopyAssign& operator=(const CopyAssign&) = default;

#if TEST_STD_VER >= 2020 || !TEST_COMPILER(MSVC)
  TEST_FUNC constexpr const CopyAssign& operator=(const CopyAssign&) const = delete;
#endif // TEST_STD_VER >= 2020 || !TEST_COMPILER(MSVC)
  TEST_FUNC constexpr CopyAssign& operator=(CopyAssign&&)             = delete;
  TEST_FUNC constexpr const CopyAssign& operator=(CopyAssign&&) const = delete;
};

struct ConstCopyAssign
{
  mutable int val{};

  constexpr ConstCopyAssign() = default;
  TEST_FUNC constexpr ConstCopyAssign(int v)
      : val(v)
  {}

  TEST_FUNC constexpr const ConstCopyAssign& operator=( // NOLINT(misc-unconventional-assign-operator)
    const ConstCopyAssign& other) const
  {
    val = other.val;
    return *this;
  }

  TEST_FUNC constexpr ConstCopyAssign& operator=(const ConstCopyAssign&)        = delete;
  TEST_FUNC constexpr ConstCopyAssign& operator=(ConstCopyAssign&&)             = delete;
  TEST_FUNC constexpr const ConstCopyAssign& operator=(ConstCopyAssign&&) const = delete;
};

struct MoveAssign
{
  int val{};

  constexpr MoveAssign() = default;
  TEST_FUNC constexpr MoveAssign(int v)
      : val(v)
  {}

  constexpr MoveAssign& operator=(MoveAssign&&) = default;

  TEST_FUNC constexpr MoveAssign& operator=(const MoveAssign&)             = delete;
  TEST_FUNC constexpr const MoveAssign& operator=(const MoveAssign&) const = delete;
#if TEST_STD_VER >= 2020 || !TEST_COMPILER(MSVC)
  TEST_FUNC constexpr const MoveAssign& operator=(MoveAssign&&) const = delete;
#endif // TEST_STD_VER >= 2020 || !TEST_COMPILER(MSVC)
};

struct ConstMoveAssign
{
  mutable int val{};

  constexpr ConstMoveAssign() = default;
  TEST_FUNC constexpr ConstMoveAssign(int v)
      : val(v)
  {}

  // NOLINTNEXTLINE(misc-unconventional-assign-operator)
  TEST_FUNC constexpr const ConstMoveAssign& operator=(ConstMoveAssign&& other) const noexcept
  {
    val = other.val;
    return *this;
  }

  TEST_FUNC constexpr ConstMoveAssign& operator=(const ConstMoveAssign&)             = delete;
  TEST_FUNC constexpr const ConstMoveAssign& operator=(const ConstMoveAssign&) const = delete;
  TEST_FUNC constexpr ConstMoveAssign& operator=(ConstMoveAssign&&)                  = delete;
};

template <class T>
struct AssignableFrom
{
  T v{};

  constexpr AssignableFrom() = default;

  // NOLINTBEGIN(bugprone-forwarding-reference-overload)
  template <class U, cuda::std::enable_if_t<cuda::std::is_constructible_v<T, U&&>, int> = 0>
  TEST_FUNC constexpr AssignableFrom(U&& u)
      : v(cuda::std::forward<U>(u))
  {}
  // NOLINTEND(bugprone-forwarding-reference-overload)

  template <class U = T, cuda::std::enable_if_t<cuda::std::is_copy_assignable_v<U>, int> = 0>
  TEST_FUNC constexpr AssignableFrom& operator=(const T& t)
  {
    v = t;
    return *this;
  }

  template <class U = T, cuda::std::enable_if_t<cuda::std::is_move_assignable_v<U>, int> = 0>
  TEST_FUNC constexpr AssignableFrom& operator=(T&& t)
  {
    v = cuda::std::move(t);
    return *this;
  }

  template <class U = T, cuda::std::enable_if_t<cuda::std::is_assignable_v<const U&, const U&>, int> = 0>
  TEST_FUNC constexpr const AssignableFrom& operator=( // NOLINT(misc-unconventional-assign-operator)
    const T& t) const
  {
    v = t;
    return *this;
  }

  template <class U = T, cuda::std::enable_if_t<cuda::std::is_assignable_v<const U&, U&&>, int> = 0>
  TEST_FUNC constexpr const AssignableFrom& operator=( // NOLINT(misc-unconventional-assign-operator)
    T&& t) const
  {
    v = cuda::std::move(t);
    return *this;
  }
};

struct TracedAssignment
{
  int copyAssign              = 0;
  mutable int constCopyAssign = 0;
  int moveAssign              = 0;
  mutable int constMoveAssign = 0;

  constexpr TracedAssignment() = default;

  TEST_FUNC constexpr TracedAssignment& operator=(const TracedAssignment&)
  {
    copyAssign++;
    return *this;
  }
  TEST_FUNC constexpr const TracedAssignment& operator=( // NOLINT(misc-unconventional-assign-operator)
    const TracedAssignment&) const
  {
    constCopyAssign++;
    return *this;
  }
  TEST_FUNC constexpr TracedAssignment& operator=(TracedAssignment&&) noexcept
  {
    moveAssign++;
    return *this;
  }
  TEST_FUNC constexpr const TracedAssignment& operator=( // NOLINT(misc-unconventional-assign-operator)
    TracedAssignment&&) const noexcept
  {
    constMoveAssign++;
    return *this;
  }
};
#endif // TEST_STD_UTILITIES_TUPLE_CNSTR_TYPES_H
