//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef TEST_SUPPORT_COMPARE_TYPES_H
#define TEST_SUPPORT_COMPARE_TYPES_H

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "test_macros.h"

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <compare>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

// `noexcept` specifiers deliberately imperfect since not all programmers bother to put the
// specifiers on their overloads.

struct equality_comparable_with_ec1;
struct no_neq;

#if TEST_STD_VER > 2017
struct cxx20_member_eq
{
  bool operator==(cxx20_member_eq const&) const = default;
};

struct cxx20_friend_eq
{
  TEST_FUNC friend bool operator==(cxx20_friend_eq const&, cxx20_friend_eq const&) = default;
};

#  if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
struct member_three_way_comparable
{
  auto operator<=>(member_three_way_comparable const&) const = default;
};

#    if !TEST_CUDA_COMPILER(NVCC) // nvbug3908399
struct friend_three_way_comparable
{
  TEST_FUNC friend auto operator<=>(friend_three_way_comparable const&, friend_three_way_comparable const&) = default;
};
#    endif // !TEST_CUDA_COMPILER(NVCC)
#  endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#endif // TEST_STD_VER > 2017

struct explicit_operators
{
  TEST_FUNC friend bool operator==(explicit_operators, explicit_operators) noexcept;
  TEST_FUNC friend bool operator!=(explicit_operators, explicit_operators) noexcept;
  TEST_FUNC friend bool operator<(explicit_operators, explicit_operators) noexcept;
  TEST_FUNC friend bool operator>(explicit_operators, explicit_operators) noexcept;
  TEST_FUNC friend bool operator<=(explicit_operators, explicit_operators) noexcept;
  TEST_FUNC friend bool operator>=(explicit_operators, explicit_operators) noexcept;

  TEST_FUNC friend bool operator==(explicit_operators const&, equality_comparable_with_ec1 const&) noexcept;
  TEST_FUNC friend bool operator==(equality_comparable_with_ec1 const&, explicit_operators const&) noexcept;
  TEST_FUNC friend bool operator!=(explicit_operators const&, equality_comparable_with_ec1 const&) noexcept;
  TEST_FUNC friend bool operator!=(equality_comparable_with_ec1 const&, explicit_operators const&) noexcept;
};

struct different_return_types
{
  TEST_FUNC bool operator==(different_return_types) const noexcept;
  TEST_FUNC char operator!=(different_return_types) const noexcept;
  TEST_FUNC short operator<(different_return_types) const noexcept;
  TEST_FUNC int operator>(different_return_types) const noexcept;
  TEST_FUNC long operator<=(different_return_types) const noexcept;
  TEST_FUNC long long operator>=(different_return_types) const noexcept;

  TEST_FUNC friend signed char operator==(explicit_operators, different_return_types);
  TEST_FUNC friend unsigned char operator==(different_return_types, explicit_operators);
  TEST_FUNC friend float operator!=(explicit_operators, different_return_types);
  TEST_FUNC friend double operator!=(different_return_types, explicit_operators);

  TEST_FUNC operator explicit_operators() const;
};

struct boolean
{
  TEST_FUNC operator bool() const noexcept;
};

struct one_member_one_friend
{
  TEST_FUNC friend boolean operator==(one_member_one_friend, one_member_one_friend) noexcept;
  TEST_FUNC boolean operator!=(one_member_one_friend) const noexcept;

  TEST_FUNC operator explicit_operators() const noexcept;
  TEST_FUNC operator different_return_types() const noexcept;
};

struct equality_comparable_with_ec1
{
  TEST_FUNC bool operator==(equality_comparable_with_ec1) const noexcept;
  TEST_FUNC bool operator!=(equality_comparable_with_ec1) const noexcept;
  TEST_FUNC operator explicit_operators() const noexcept;
};

struct no_eq
{
  TEST_FUNC friend bool operator==(no_eq, no_eq) = delete;
  TEST_FUNC friend bool operator!=(no_eq, no_eq) noexcept;
  TEST_FUNC friend bool operator<(no_eq, no_eq) noexcept;
  TEST_FUNC friend bool operator>(no_eq, no_eq) noexcept;
  TEST_FUNC friend bool operator>=(no_eq, no_eq) noexcept;
  TEST_FUNC friend bool operator<=(no_eq, no_eq) noexcept;
};

struct no_neq
{
  TEST_FUNC friend bool operator==(no_neq, no_neq) noexcept;
  TEST_FUNC friend bool operator!=(no_neq, no_neq) = delete;
  TEST_FUNC friend bool operator<(no_eq, no_eq) noexcept;
  TEST_FUNC friend bool operator>(no_eq, no_eq) noexcept;
  TEST_FUNC friend bool operator>=(no_eq, no_eq) noexcept;
  TEST_FUNC friend bool operator<=(no_eq, no_eq) noexcept;
};

struct no_lt
{
  TEST_FUNC friend bool operator==(no_lt, no_lt) noexcept;
  TEST_FUNC friend bool operator!=(no_lt, no_lt) noexcept;
  TEST_FUNC friend bool operator<(no_lt, no_lt) = delete;
  TEST_FUNC friend bool operator>(no_lt, no_lt) noexcept;
  TEST_FUNC friend bool operator>=(no_lt, no_lt) noexcept;
  TEST_FUNC friend bool operator<=(no_lt, no_lt) noexcept;
};

struct no_gt
{
  TEST_FUNC friend bool operator==(no_gt, no_gt) noexcept;
  TEST_FUNC friend bool operator!=(no_gt, no_gt) noexcept;
  TEST_FUNC friend bool operator<(no_gt, no_gt) noexcept;
  TEST_FUNC friend bool operator>(no_gt, no_gt) = delete;
  TEST_FUNC friend bool operator>=(no_gt, no_gt) noexcept;
  TEST_FUNC friend bool operator<=(no_gt, no_gt) noexcept;
};

struct no_le
{
  TEST_FUNC friend bool operator==(no_le, no_le) noexcept;
  TEST_FUNC friend bool operator!=(no_le, no_le) noexcept;
  TEST_FUNC friend bool operator<(no_le, no_le) noexcept;
  TEST_FUNC friend bool operator>(no_le, no_le) noexcept;
  TEST_FUNC friend bool operator>=(no_le, no_le) = delete;
  TEST_FUNC friend bool operator<=(no_le, no_le) noexcept;
};

struct no_ge
{
  TEST_FUNC friend bool operator==(no_ge, no_ge) noexcept;
  TEST_FUNC friend bool operator!=(no_ge, no_ge) noexcept;
  TEST_FUNC friend bool operator<(no_ge, no_ge) noexcept;
  TEST_FUNC friend bool operator>(no_ge, no_ge) noexcept;
  TEST_FUNC friend bool operator>=(no_ge, no_ge) noexcept;
  TEST_FUNC friend bool operator<=(no_ge, no_ge) = delete;
};

struct wrong_return_type_eq
{
  TEST_FUNC void operator==(wrong_return_type_eq) const noexcept;
  TEST_FUNC bool operator!=(wrong_return_type_eq) const noexcept;
  TEST_FUNC bool operator<(wrong_return_type_eq) const noexcept;
  TEST_FUNC bool operator>(wrong_return_type_eq) const noexcept;
  TEST_FUNC bool operator>=(wrong_return_type_eq) const noexcept;
  TEST_FUNC bool operator<=(wrong_return_type_eq) const noexcept;
};

struct wrong_return_type_ne
{
  TEST_FUNC bool operator==(wrong_return_type_ne) const noexcept;
  TEST_FUNC void operator!=(wrong_return_type_ne) const noexcept;
  TEST_FUNC bool operator<(wrong_return_type_ne) const noexcept;
  TEST_FUNC bool operator>(wrong_return_type_ne) const noexcept;
  TEST_FUNC bool operator>=(wrong_return_type_ne) const noexcept;
  TEST_FUNC bool operator<=(wrong_return_type_ne) const noexcept;
};

struct wrong_return_type_lt
{
  TEST_FUNC bool operator==(wrong_return_type_lt) const noexcept;
  TEST_FUNC bool operator!=(wrong_return_type_lt) const noexcept;
  TEST_FUNC void operator<(wrong_return_type_lt) const noexcept;
  TEST_FUNC bool operator>(wrong_return_type_lt) const noexcept;
  TEST_FUNC bool operator>=(wrong_return_type_lt) const noexcept;
  TEST_FUNC bool operator<=(wrong_return_type_lt) const noexcept;
};

struct wrong_return_type_gt
{
  TEST_FUNC bool operator==(wrong_return_type_gt) const noexcept;
  TEST_FUNC bool operator!=(wrong_return_type_gt) const noexcept;
  TEST_FUNC bool operator<(wrong_return_type_gt) const noexcept;
  TEST_FUNC void operator>(wrong_return_type_gt) const noexcept;
  TEST_FUNC bool operator>=(wrong_return_type_gt) const noexcept;
  TEST_FUNC bool operator<=(wrong_return_type_gt) const noexcept;
};

struct wrong_return_type_le
{
  TEST_FUNC bool operator==(wrong_return_type_le) const noexcept;
  TEST_FUNC bool operator!=(wrong_return_type_le) const noexcept;
  TEST_FUNC bool operator<(wrong_return_type_le) const noexcept;
  TEST_FUNC bool operator>(wrong_return_type_le) const noexcept;
  TEST_FUNC void operator>=(wrong_return_type_le) const noexcept;
  TEST_FUNC bool operator<=(wrong_return_type_le) const noexcept;
};

struct wrong_return_type_ge
{
  TEST_FUNC bool operator==(wrong_return_type_ge) const noexcept;
  TEST_FUNC bool operator!=(wrong_return_type_ge) const noexcept;
  TEST_FUNC bool operator<(wrong_return_type_ge) const noexcept;
  TEST_FUNC bool operator>(wrong_return_type_ge) const noexcept;
  TEST_FUNC bool operator>=(wrong_return_type_ge) const noexcept;
  TEST_FUNC void operator<=(wrong_return_type_ge) const noexcept;
};

struct wrong_return_type
{
  TEST_FUNC void operator==(wrong_return_type) const noexcept;
  TEST_FUNC void operator!=(wrong_return_type) const noexcept;
  TEST_FUNC void operator<(wrong_return_type) const noexcept;
  TEST_FUNC void operator>(wrong_return_type) const noexcept;
  TEST_FUNC void operator>=(wrong_return_type) const noexcept;
  TEST_FUNC void operator<=(wrong_return_type_ge) const noexcept;
};

#if TEST_STD_VER > 2017
struct cxx20_member_eq_operator_with_deleted_ne
{
  bool operator==(cxx20_member_eq_operator_with_deleted_ne const&) const           = default;
  TEST_FUNC bool operator!=(cxx20_member_eq_operator_with_deleted_ne const&) const = delete;
};

struct cxx20_friend_eq_operator_with_deleted_ne
{
  TEST_FUNC friend bool operator==(cxx20_friend_eq_operator_with_deleted_ne const&,
                                   cxx20_friend_eq_operator_with_deleted_ne const&) = default;
  TEST_FUNC friend bool operator!=(cxx20_friend_eq_operator_with_deleted_ne const&,
                                   cxx20_friend_eq_operator_with_deleted_ne const&) = delete;
};

#  if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
struct member_three_way_comparable_with_deleted_eq
{
  auto operator<=>(member_three_way_comparable_with_deleted_eq const&) const          = default;
  TEST_FUNC bool operator==(member_three_way_comparable_with_deleted_eq const&) const = delete;
};

struct member_three_way_comparable_with_deleted_ne
{
  auto operator<=>(member_three_way_comparable_with_deleted_ne const&) const          = default;
  TEST_FUNC bool operator!=(member_three_way_comparable_with_deleted_ne const&) const = delete;
};

struct friend_three_way_comparable_with_deleted_eq
{
  TEST_FUNC friend auto operator<=>(friend_three_way_comparable_with_deleted_eq const&,
                                    friend_three_way_comparable_with_deleted_eq const&) = default;
  TEST_FUNC friend bool operator==(friend_three_way_comparable_with_deleted_eq const&,
                                   friend_three_way_comparable_with_deleted_eq const&)  = delete;
};

#    if !TEST_CUDA_COMPILER(NVCC) // nvbug3908399
struct friend_three_way_comparable_with_deleted_ne
{
  TEST_FUNC friend auto operator<=>(friend_three_way_comparable_with_deleted_ne const&,
                                    friend_three_way_comparable_with_deleted_ne const&) = default;
  TEST_FUNC friend bool operator!=(friend_three_way_comparable_with_deleted_ne const&,
                                   friend_three_way_comparable_with_deleted_ne const&)  = delete;
};
#    endif // !TEST_CUDA_COMPILER(NVCC)
#  endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

struct one_way_eq
{
  bool operator==(one_way_eq const&) const = default;
  TEST_FUNC friend bool operator==(one_way_eq, explicit_operators);
  TEST_FUNC friend bool operator==(explicit_operators, one_way_eq) = delete;

  TEST_FUNC operator explicit_operators() const;
};

struct one_way_ne
{
  bool operator==(one_way_ne const&) const = default;
  TEST_FUNC friend bool operator==(one_way_ne, explicit_operators);
  TEST_FUNC friend bool operator!=(one_way_ne, explicit_operators) = delete;

  TEST_FUNC operator explicit_operators() const;
};
static_assert(requires(explicit_operators const x, one_way_ne const y) { x != y; });

struct explicit_bool
{
  TEST_FUNC explicit operator bool() const noexcept;
};

#  if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
struct totally_ordered_with_others
{
  auto operator<=>(totally_ordered_with_others const&) const = default;
};

struct no_lt_not_totally_ordered_with
{
  bool operator==(no_lt_not_totally_ordered_with const&) const  = default;
  auto operator<=>(no_lt_not_totally_ordered_with const&) const = default;
  TEST_FUNC operator totally_ordered_with_others() const noexcept;

  TEST_FUNC bool operator==(totally_ordered_with_others const&) const;
  TEST_FUNC auto operator<=>(totally_ordered_with_others const&) const;
  TEST_FUNC auto operator<(totally_ordered_with_others const&) const;
};

struct no_gt_not_totally_ordered_with
{
  bool operator==(no_gt_not_totally_ordered_with const&) const  = default;
  auto operator<=>(no_gt_not_totally_ordered_with const&) const = default;
  TEST_FUNC operator totally_ordered_with_others() const noexcept;

  TEST_FUNC bool operator==(totally_ordered_with_others const&) const;
  TEST_FUNC auto operator<=>(totally_ordered_with_others const&) const;
  TEST_FUNC auto operator>(totally_ordered_with_others const&) const;
};

struct no_le_not_totally_ordered_with
{
  bool operator==(no_le_not_totally_ordered_with const&) const  = default;
  auto operator<=>(no_le_not_totally_ordered_with const&) const = default;
  TEST_FUNC operator totally_ordered_with_others() const noexcept;

  TEST_FUNC bool operator==(totally_ordered_with_others const&) const;
  TEST_FUNC auto operator<=>(totally_ordered_with_others const&) const;
  TEST_FUNC auto operator<=(totally_ordered_with_others const&) const;
};

struct no_ge_not_totally_ordered_with
{
  bool operator==(no_ge_not_totally_ordered_with const&) const  = default;
  auto operator<=>(no_ge_not_totally_ordered_with const&) const = default;
  TEST_FUNC operator totally_ordered_with_others() const noexcept;

  TEST_FUNC bool operator==(totally_ordered_with_others const&) const;
  TEST_FUNC auto operator<=>(totally_ordered_with_others const&) const;
  TEST_FUNC auto operator>=(totally_ordered_with_others const&) const;
};

struct partial_ordering_totally_ordered_with
{
  auto operator<=>(partial_ordering_totally_ordered_with const&) const noexcept = default;
  TEST_FUNC std::partial_ordering operator<=>(totally_ordered_with_others const&) const noexcept;

  TEST_FUNC operator totally_ordered_with_others() const;
};

struct weak_ordering_totally_ordered_with
{
  auto operator<=>(weak_ordering_totally_ordered_with const&) const noexcept = default;
  TEST_FUNC std::weak_ordering operator<=>(totally_ordered_with_others const&) const noexcept;

  TEST_FUNC operator totally_ordered_with_others() const;
};

struct strong_ordering_totally_ordered_with
{
  auto operator<=>(strong_ordering_totally_ordered_with const&) const noexcept = default;
  TEST_FUNC std::strong_ordering operator<=>(totally_ordered_with_others const&) const noexcept;

  TEST_FUNC operator totally_ordered_with_others() const;
};

struct eq_returns_explicit_bool
{
  TEST_FUNC friend explicit_bool operator==(eq_returns_explicit_bool, eq_returns_explicit_bool);
  TEST_FUNC friend bool operator!=(eq_returns_explicit_bool, eq_returns_explicit_bool);
  TEST_FUNC friend bool operator<(eq_returns_explicit_bool, eq_returns_explicit_bool);
  TEST_FUNC friend bool operator>(eq_returns_explicit_bool, eq_returns_explicit_bool);
  TEST_FUNC friend bool operator<=(eq_returns_explicit_bool, eq_returns_explicit_bool);
  TEST_FUNC friend bool operator>=(eq_returns_explicit_bool, eq_returns_explicit_bool);

  TEST_FUNC operator totally_ordered_with_others() const;

  TEST_FUNC friend explicit_bool operator==(eq_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend explicit_bool operator==(totally_ordered_with_others, eq_returns_explicit_bool);
  TEST_FUNC friend bool operator!=(eq_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator!=(totally_ordered_with_others, eq_returns_explicit_bool);
  TEST_FUNC friend bool operator<(eq_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator<(totally_ordered_with_others, eq_returns_explicit_bool);
  TEST_FUNC friend bool operator>(eq_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator>(totally_ordered_with_others, eq_returns_explicit_bool);
  TEST_FUNC friend bool operator<=(eq_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator<=(totally_ordered_with_others, eq_returns_explicit_bool);
  TEST_FUNC friend bool operator>=(eq_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator>=(totally_ordered_with_others, eq_returns_explicit_bool);
};

struct ne_returns_explicit_bool
{
  TEST_FUNC friend bool operator==(ne_returns_explicit_bool, ne_returns_explicit_bool);
  TEST_FUNC friend explicit_bool operator!=(ne_returns_explicit_bool, ne_returns_explicit_bool);
  TEST_FUNC friend bool operator<(ne_returns_explicit_bool, ne_returns_explicit_bool);
  TEST_FUNC friend bool operator>(ne_returns_explicit_bool, ne_returns_explicit_bool);
  TEST_FUNC friend bool operator<=(ne_returns_explicit_bool, ne_returns_explicit_bool);
  TEST_FUNC friend bool operator>=(ne_returns_explicit_bool, ne_returns_explicit_bool);

  TEST_FUNC operator totally_ordered_with_others() const;

  TEST_FUNC friend bool operator==(ne_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend explicit_bool operator!=(ne_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend explicit_bool operator!=(totally_ordered_with_others, ne_returns_explicit_bool);
  TEST_FUNC friend bool operator<(ne_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator<(totally_ordered_with_others, ne_returns_explicit_bool);
  TEST_FUNC friend bool operator>(ne_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator>(totally_ordered_with_others, ne_returns_explicit_bool);
  TEST_FUNC friend bool operator<=(ne_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator<=(totally_ordered_with_others, ne_returns_explicit_bool);
  TEST_FUNC friend bool operator>=(ne_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator>=(totally_ordered_with_others, ne_returns_explicit_bool);
};

struct lt_returns_explicit_bool
{
  TEST_FUNC friend bool operator==(lt_returns_explicit_bool, lt_returns_explicit_bool);
  TEST_FUNC friend bool operator!=(lt_returns_explicit_bool, lt_returns_explicit_bool);
  TEST_FUNC friend explicit_bool operator<(lt_returns_explicit_bool, lt_returns_explicit_bool);
  TEST_FUNC friend bool operator>(lt_returns_explicit_bool, lt_returns_explicit_bool);
  TEST_FUNC friend bool operator<=(lt_returns_explicit_bool, lt_returns_explicit_bool);
  TEST_FUNC friend bool operator>=(lt_returns_explicit_bool, lt_returns_explicit_bool);

  TEST_FUNC operator totally_ordered_with_others() const;

  TEST_FUNC friend bool operator==(lt_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator!=(lt_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator!=(totally_ordered_with_others, lt_returns_explicit_bool);
  TEST_FUNC friend explicit_bool operator<(lt_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator<(totally_ordered_with_others, lt_returns_explicit_bool);
  TEST_FUNC friend bool operator>(lt_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator>(totally_ordered_with_others, lt_returns_explicit_bool);
  TEST_FUNC friend bool operator<=(lt_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator<=(totally_ordered_with_others, lt_returns_explicit_bool);
  TEST_FUNC friend bool operator>=(lt_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator>=(totally_ordered_with_others, lt_returns_explicit_bool);
};

struct gt_returns_explicit_bool
{
  TEST_FUNC friend bool operator==(gt_returns_explicit_bool, gt_returns_explicit_bool);
  TEST_FUNC friend bool operator!=(gt_returns_explicit_bool, gt_returns_explicit_bool);
  TEST_FUNC friend bool operator<(gt_returns_explicit_bool, gt_returns_explicit_bool);
  TEST_FUNC friend explicit_bool operator>(gt_returns_explicit_bool, gt_returns_explicit_bool);
  TEST_FUNC friend bool operator<=(gt_returns_explicit_bool, gt_returns_explicit_bool);
  TEST_FUNC friend bool operator>=(gt_returns_explicit_bool, gt_returns_explicit_bool);

  TEST_FUNC operator totally_ordered_with_others() const;

  TEST_FUNC friend bool operator==(gt_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator!=(gt_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator!=(totally_ordered_with_others, gt_returns_explicit_bool);
  TEST_FUNC friend bool operator<(gt_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator<(totally_ordered_with_others, gt_returns_explicit_bool);
  TEST_FUNC friend explicit_bool operator>(gt_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator>(totally_ordered_with_others, gt_returns_explicit_bool);
  TEST_FUNC friend bool operator<=(gt_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator<=(totally_ordered_with_others, gt_returns_explicit_bool);
  TEST_FUNC friend bool operator>=(gt_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator>=(totally_ordered_with_others, gt_returns_explicit_bool);
};

struct le_returns_explicit_bool
{
  TEST_FUNC friend bool operator==(le_returns_explicit_bool, le_returns_explicit_bool);
  TEST_FUNC friend bool operator!=(le_returns_explicit_bool, le_returns_explicit_bool);
  TEST_FUNC friend bool operator<(le_returns_explicit_bool, le_returns_explicit_bool);
  TEST_FUNC friend bool operator>(le_returns_explicit_bool, le_returns_explicit_bool);
  TEST_FUNC friend explicit_bool operator<=(le_returns_explicit_bool, le_returns_explicit_bool);
  TEST_FUNC friend bool operator>=(le_returns_explicit_bool, le_returns_explicit_bool);

  TEST_FUNC operator totally_ordered_with_others() const;

  TEST_FUNC friend bool operator==(le_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator!=(le_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator!=(totally_ordered_with_others, le_returns_explicit_bool);
  TEST_FUNC friend bool operator<(le_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator<(totally_ordered_with_others, le_returns_explicit_bool);
  TEST_FUNC friend bool operator>(le_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator>(totally_ordered_with_others, le_returns_explicit_bool);
  TEST_FUNC friend bool operator<=(le_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend explicit_bool operator<=(totally_ordered_with_others, le_returns_explicit_bool);
  TEST_FUNC friend bool operator>=(le_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator>=(totally_ordered_with_others, le_returns_explicit_bool);
};

struct ge_returns_explicit_bool
{
  TEST_FUNC friend bool operator==(ge_returns_explicit_bool, ge_returns_explicit_bool);
  TEST_FUNC friend bool operator!=(ge_returns_explicit_bool, ge_returns_explicit_bool);
  TEST_FUNC friend bool operator<(ge_returns_explicit_bool, ge_returns_explicit_bool);
  TEST_FUNC friend bool operator>(ge_returns_explicit_bool, ge_returns_explicit_bool);
  TEST_FUNC friend bool operator<=(ge_returns_explicit_bool, ge_returns_explicit_bool);
  TEST_FUNC friend explicit_bool operator>=(ge_returns_explicit_bool, ge_returns_explicit_bool);

  TEST_FUNC operator totally_ordered_with_others() const;

  TEST_FUNC friend bool operator==(ge_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator!=(ge_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator!=(totally_ordered_with_others, ge_returns_explicit_bool);
  TEST_FUNC friend bool operator<(ge_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator<(totally_ordered_with_others, ge_returns_explicit_bool);
  TEST_FUNC friend bool operator>(ge_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator>(totally_ordered_with_others, ge_returns_explicit_bool);
  TEST_FUNC friend bool operator<=(ge_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend bool operator<=(totally_ordered_with_others, ge_returns_explicit_bool);
  TEST_FUNC friend bool operator>=(ge_returns_explicit_bool, totally_ordered_with_others);
  TEST_FUNC friend explicit_bool operator>=(totally_ordered_with_others, ge_returns_explicit_bool);
};

struct returns_true_type
{
  TEST_FUNC friend cuda::std::true_type operator==(returns_true_type, returns_true_type);
  TEST_FUNC friend cuda::std::true_type operator!=(returns_true_type, returns_true_type);
  TEST_FUNC friend cuda::std::true_type operator<(returns_true_type, returns_true_type);
  TEST_FUNC friend cuda::std::true_type operator>(returns_true_type, returns_true_type);
  TEST_FUNC friend cuda::std::true_type operator<=(returns_true_type, returns_true_type);
  TEST_FUNC friend cuda::std::true_type operator>=(returns_true_type, returns_true_type);

  TEST_FUNC operator totally_ordered_with_others() const;

  TEST_FUNC friend cuda::std::true_type operator==(returns_true_type, totally_ordered_with_others);
  TEST_FUNC friend cuda::std::true_type operator==(totally_ordered_with_others, returns_true_type);
  TEST_FUNC friend cuda::std::true_type operator!=(returns_true_type, totally_ordered_with_others);
  TEST_FUNC friend cuda::std::true_type operator!=(totally_ordered_with_others, returns_true_type);
  TEST_FUNC friend cuda::std::true_type operator<(returns_true_type, totally_ordered_with_others);
  TEST_FUNC friend cuda::std::true_type operator<(totally_ordered_with_others, returns_true_type);
  TEST_FUNC friend cuda::std::true_type operator>(returns_true_type, totally_ordered_with_others);
  TEST_FUNC friend cuda::std::true_type operator>(totally_ordered_with_others, returns_true_type);
  TEST_FUNC friend cuda::std::true_type operator<=(returns_true_type, totally_ordered_with_others);
  TEST_FUNC friend cuda::std::true_type operator<=(totally_ordered_with_others, returns_true_type);
  TEST_FUNC friend cuda::std::true_type operator>=(returns_true_type, totally_ordered_with_others);
  TEST_FUNC friend cuda::std::true_type operator>=(totally_ordered_with_others, returns_true_type);
};

struct returns_int_ptr
{
  TEST_FUNC friend int* operator==(returns_int_ptr, returns_int_ptr);
  TEST_FUNC friend int* operator!=(returns_int_ptr, returns_int_ptr);
  TEST_FUNC friend int* operator<(returns_int_ptr, returns_int_ptr);
  TEST_FUNC friend int* operator>(returns_int_ptr, returns_int_ptr);
  TEST_FUNC friend int* operator<=(returns_int_ptr, returns_int_ptr);
  TEST_FUNC friend int* operator>=(returns_int_ptr, returns_int_ptr);

  TEST_FUNC operator totally_ordered_with_others() const;

  TEST_FUNC friend int* operator==(returns_int_ptr, totally_ordered_with_others);
  TEST_FUNC friend int* operator==(totally_ordered_with_others, returns_int_ptr);
  TEST_FUNC friend int* operator!=(returns_int_ptr, totally_ordered_with_others);
  TEST_FUNC friend int* operator!=(totally_ordered_with_others, returns_int_ptr);
  TEST_FUNC friend int* operator<(returns_int_ptr, totally_ordered_with_others);
  TEST_FUNC friend int* operator<(totally_ordered_with_others, returns_int_ptr);
  TEST_FUNC friend int* operator>(returns_int_ptr, totally_ordered_with_others);
  TEST_FUNC friend int* operator>(totally_ordered_with_others, returns_int_ptr);
  TEST_FUNC friend int* operator<=(returns_int_ptr, totally_ordered_with_others);
  TEST_FUNC friend int* operator<=(totally_ordered_with_others, returns_int_ptr);
  TEST_FUNC friend int* operator>=(returns_int_ptr, totally_ordered_with_others);
  TEST_FUNC friend int* operator>=(totally_ordered_with_others, returns_int_ptr);
};
#  endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#endif // TEST_STD_VER > 2017

struct ForwardingTestObject
{
  TEST_FUNC constexpr bool operator<(ForwardingTestObject&&) &&
  {
    return true;
  }
  TEST_FUNC constexpr bool operator<(const ForwardingTestObject&) const&
  {
    return false;
  }

  TEST_FUNC constexpr bool operator==(ForwardingTestObject&&) &&
  {
    return true;
  }
  TEST_FUNC constexpr bool operator==(const ForwardingTestObject&) const&
  {
    return false;
  }

  TEST_FUNC constexpr bool operator!=(ForwardingTestObject&&) &&
  {
    return true;
  }
  TEST_FUNC constexpr bool operator!=(const ForwardingTestObject&) const&
  {
    return false;
  }

  TEST_FUNC constexpr bool operator<=(ForwardingTestObject&&) &&
  {
    return true;
  }
  TEST_FUNC constexpr bool operator<=(const ForwardingTestObject&) const&
  {
    return false;
  }

  TEST_FUNC constexpr bool operator>(ForwardingTestObject&&) &&
  {
    return true;
  }
  TEST_FUNC constexpr bool operator>(const ForwardingTestObject&) const&
  {
    return false;
  }

  TEST_FUNC constexpr bool operator>=(ForwardingTestObject&&) &&
  {
    return true;
  }
  TEST_FUNC constexpr bool operator>=(const ForwardingTestObject&) const&
  {
    return false;
  }
};

#endif // TEST_SUPPORT_COMPARE_TYPES_H
