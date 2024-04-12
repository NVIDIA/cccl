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

#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
#  include <compare>
#endif // TEST_HAS_NO_SPACESHIP_OPERATOR

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
  __host__ __device__ friend bool operator==(cxx20_friend_eq const&, cxx20_friend_eq const&) = default;
};

#  ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
struct member_three_way_comparable
{
  auto operator<=>(member_three_way_comparable const&) const = default;
};

#    ifndef __NVCC__ // nvbug3908399
struct friend_three_way_comparable
{
  __host__ __device__ friend auto
  operator<=>(friend_three_way_comparable const&, friend_three_way_comparable const&) = default;
};
#    endif // !__NVCC__
#  endif // TEST_HAS_NO_SPACESHIP_OPERATOR
#endif // TEST_STD_VER > 2017

struct explicit_operators
{
  __host__ __device__ friend bool operator==(explicit_operators, explicit_operators) noexcept;
  __host__ __device__ friend bool operator!=(explicit_operators, explicit_operators) noexcept;
  __host__ __device__ friend bool operator<(explicit_operators, explicit_operators) noexcept;
  __host__ __device__ friend bool operator>(explicit_operators, explicit_operators) noexcept;
  __host__ __device__ friend bool operator<=(explicit_operators, explicit_operators) noexcept;
  __host__ __device__ friend bool operator>=(explicit_operators, explicit_operators) noexcept;

  __host__ __device__ friend bool operator==(explicit_operators const&, equality_comparable_with_ec1 const&) noexcept;
  __host__ __device__ friend bool operator==(equality_comparable_with_ec1 const&, explicit_operators const&) noexcept;
  __host__ __device__ friend bool operator!=(explicit_operators const&, equality_comparable_with_ec1 const&) noexcept;
  __host__ __device__ friend bool operator!=(equality_comparable_with_ec1 const&, explicit_operators const&) noexcept;
};

struct different_return_types
{
  __host__ __device__ bool operator==(different_return_types) const noexcept;
  __host__ __device__ char operator!=(different_return_types) const noexcept;
  __host__ __device__ short operator<(different_return_types) const noexcept;
  __host__ __device__ int operator>(different_return_types) const noexcept;
  __host__ __device__ long operator<=(different_return_types) const noexcept;
  __host__ __device__ long long operator>=(different_return_types) const noexcept;

  __host__ __device__ friend signed char operator==(explicit_operators, different_return_types);
  __host__ __device__ friend unsigned char operator==(different_return_types, explicit_operators);
  __host__ __device__ friend float operator!=(explicit_operators, different_return_types);
  __host__ __device__ friend double operator!=(different_return_types, explicit_operators);

  __host__ __device__ operator explicit_operators() const;
};

struct boolean
{
  __host__ __device__ operator bool() const noexcept;
};

struct one_member_one_friend
{
  __host__ __device__ friend boolean operator==(one_member_one_friend, one_member_one_friend) noexcept;
  __host__ __device__ boolean operator!=(one_member_one_friend) const noexcept;

  __host__ __device__ operator explicit_operators() const noexcept;
  __host__ __device__ operator different_return_types() const noexcept;
};

struct equality_comparable_with_ec1
{
  __host__ __device__ bool operator==(equality_comparable_with_ec1) const noexcept;
  __host__ __device__ bool operator!=(equality_comparable_with_ec1) const noexcept;
  __host__ __device__ operator explicit_operators() const noexcept;
};

struct no_eq
{
  __host__ __device__ friend bool operator==(no_eq, no_eq) = delete;
  __host__ __device__ friend bool operator!=(no_eq, no_eq) noexcept;
  __host__ __device__ friend bool operator<(no_eq, no_eq) noexcept;
  __host__ __device__ friend bool operator>(no_eq, no_eq) noexcept;
  __host__ __device__ friend bool operator>=(no_eq, no_eq) noexcept;
  __host__ __device__ friend bool operator<=(no_eq, no_eq) noexcept;
};

struct no_neq
{
  __host__ __device__ friend bool operator==(no_neq, no_neq) noexcept;
  __host__ __device__ friend bool operator!=(no_neq, no_neq) = delete;
  __host__ __device__ friend bool operator<(no_eq, no_eq) noexcept;
  __host__ __device__ friend bool operator>(no_eq, no_eq) noexcept;
  __host__ __device__ friend bool operator>=(no_eq, no_eq) noexcept;
  __host__ __device__ friend bool operator<=(no_eq, no_eq) noexcept;
};

struct no_lt
{
  __host__ __device__ friend bool operator==(no_lt, no_lt) noexcept;
  __host__ __device__ friend bool operator!=(no_lt, no_lt) noexcept;
  __host__ __device__ friend bool operator<(no_lt, no_lt) = delete;
  __host__ __device__ friend bool operator>(no_lt, no_lt) noexcept;
  __host__ __device__ friend bool operator>=(no_lt, no_lt) noexcept;
  __host__ __device__ friend bool operator<=(no_lt, no_lt) noexcept;
};

struct no_gt
{
  __host__ __device__ friend bool operator==(no_gt, no_gt) noexcept;
  __host__ __device__ friend bool operator!=(no_gt, no_gt) noexcept;
  __host__ __device__ friend bool operator<(no_gt, no_gt) noexcept;
  __host__ __device__ friend bool operator>(no_gt, no_gt) = delete;
  __host__ __device__ friend bool operator>=(no_gt, no_gt) noexcept;
  __host__ __device__ friend bool operator<=(no_gt, no_gt) noexcept;
};

struct no_le
{
  __host__ __device__ friend bool operator==(no_le, no_le) noexcept;
  __host__ __device__ friend bool operator!=(no_le, no_le) noexcept;
  __host__ __device__ friend bool operator<(no_le, no_le) noexcept;
  __host__ __device__ friend bool operator>(no_le, no_le) noexcept;
  __host__ __device__ friend bool operator>=(no_le, no_le) = delete;
  __host__ __device__ friend bool operator<=(no_le, no_le) noexcept;
};

struct no_ge
{
  __host__ __device__ friend bool operator==(no_ge, no_ge) noexcept;
  __host__ __device__ friend bool operator!=(no_ge, no_ge) noexcept;
  __host__ __device__ friend bool operator<(no_ge, no_ge) noexcept;
  __host__ __device__ friend bool operator>(no_ge, no_ge) noexcept;
  __host__ __device__ friend bool operator>=(no_ge, no_ge) noexcept;
  __host__ __device__ friend bool operator<=(no_ge, no_ge) = delete;
};

struct wrong_return_type_eq
{
  __host__ __device__ void operator==(wrong_return_type_eq) const noexcept;
  __host__ __device__ bool operator!=(wrong_return_type_eq) const noexcept;
  __host__ __device__ bool operator<(wrong_return_type_eq) const noexcept;
  __host__ __device__ bool operator>(wrong_return_type_eq) const noexcept;
  __host__ __device__ bool operator>=(wrong_return_type_eq) const noexcept;
  __host__ __device__ bool operator<=(wrong_return_type_eq) const noexcept;
};

struct wrong_return_type_ne
{
  __host__ __device__ bool operator==(wrong_return_type_ne) const noexcept;
  __host__ __device__ void operator!=(wrong_return_type_ne) const noexcept;
  __host__ __device__ bool operator<(wrong_return_type_ne) const noexcept;
  __host__ __device__ bool operator>(wrong_return_type_ne) const noexcept;
  __host__ __device__ bool operator>=(wrong_return_type_ne) const noexcept;
  __host__ __device__ bool operator<=(wrong_return_type_ne) const noexcept;
};

struct wrong_return_type_lt
{
  __host__ __device__ bool operator==(wrong_return_type_lt) const noexcept;
  __host__ __device__ bool operator!=(wrong_return_type_lt) const noexcept;
  __host__ __device__ void operator<(wrong_return_type_lt) const noexcept;
  __host__ __device__ bool operator>(wrong_return_type_lt) const noexcept;
  __host__ __device__ bool operator>=(wrong_return_type_lt) const noexcept;
  __host__ __device__ bool operator<=(wrong_return_type_lt) const noexcept;
};

struct wrong_return_type_gt
{
  __host__ __device__ bool operator==(wrong_return_type_gt) const noexcept;
  __host__ __device__ bool operator!=(wrong_return_type_gt) const noexcept;
  __host__ __device__ bool operator<(wrong_return_type_gt) const noexcept;
  __host__ __device__ void operator>(wrong_return_type_gt) const noexcept;
  __host__ __device__ bool operator>=(wrong_return_type_gt) const noexcept;
  __host__ __device__ bool operator<=(wrong_return_type_gt) const noexcept;
};

struct wrong_return_type_le
{
  __host__ __device__ bool operator==(wrong_return_type_le) const noexcept;
  __host__ __device__ bool operator!=(wrong_return_type_le) const noexcept;
  __host__ __device__ bool operator<(wrong_return_type_le) const noexcept;
  __host__ __device__ bool operator>(wrong_return_type_le) const noexcept;
  __host__ __device__ void operator>=(wrong_return_type_le) const noexcept;
  __host__ __device__ bool operator<=(wrong_return_type_le) const noexcept;
};

struct wrong_return_type_ge
{
  __host__ __device__ bool operator==(wrong_return_type_ge) const noexcept;
  __host__ __device__ bool operator!=(wrong_return_type_ge) const noexcept;
  __host__ __device__ bool operator<(wrong_return_type_ge) const noexcept;
  __host__ __device__ bool operator>(wrong_return_type_ge) const noexcept;
  __host__ __device__ bool operator>=(wrong_return_type_ge) const noexcept;
  __host__ __device__ void operator<=(wrong_return_type_ge) const noexcept;
};

struct wrong_return_type
{
  __host__ __device__ void operator==(wrong_return_type) const noexcept;
  __host__ __device__ void operator!=(wrong_return_type) const noexcept;
  __host__ __device__ void operator<(wrong_return_type) const noexcept;
  __host__ __device__ void operator>(wrong_return_type) const noexcept;
  __host__ __device__ void operator>=(wrong_return_type) const noexcept;
  __host__ __device__ void operator<=(wrong_return_type_ge) const noexcept;
};

#if TEST_STD_VER > 2017
struct cxx20_member_eq_operator_with_deleted_ne
{
  bool operator==(cxx20_member_eq_operator_with_deleted_ne const&) const                     = default;
  __host__ __device__ bool operator!=(cxx20_member_eq_operator_with_deleted_ne const&) const = delete;
};

struct cxx20_friend_eq_operator_with_deleted_ne
{
  __host__ __device__ friend bool operator==(cxx20_friend_eq_operator_with_deleted_ne const&,
                                             cxx20_friend_eq_operator_with_deleted_ne const&) = default;
  __host__ __device__ friend bool
  operator!=(cxx20_friend_eq_operator_with_deleted_ne const&, cxx20_friend_eq_operator_with_deleted_ne const&) = delete;
};

#  ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
struct member_three_way_comparable_with_deleted_eq
{
  auto operator<=>(member_three_way_comparable_with_deleted_eq const&) const                    = default;
  __host__ __device__ bool operator==(member_three_way_comparable_with_deleted_eq const&) const = delete;
};

struct member_three_way_comparable_with_deleted_ne
{
  auto operator<=>(member_three_way_comparable_with_deleted_ne const&) const                    = default;
  __host__ __device__ bool operator!=(member_three_way_comparable_with_deleted_ne const&) const = delete;
};

struct friend_three_way_comparable_with_deleted_eq
{
  __host__ __device__ friend auto operator<=>(friend_three_way_comparable_with_deleted_eq const&,
                                              friend_three_way_comparable_with_deleted_eq const&) = default;
  __host__ __device__ friend bool operator==(friend_three_way_comparable_with_deleted_eq const&,
                                             friend_three_way_comparable_with_deleted_eq const&)  = delete;
};

#    ifndef __NVCC__ // nvbug3908399
struct friend_three_way_comparable_with_deleted_ne
{
  __host__ __device__ friend auto operator<=>(friend_three_way_comparable_with_deleted_ne const&,
                                              friend_three_way_comparable_with_deleted_ne const&) = default;
  __host__ __device__ friend bool operator!=(friend_three_way_comparable_with_deleted_ne const&,
                                             friend_three_way_comparable_with_deleted_ne const&)  = delete;
};
#    endif // !__NVCC__
#  endif // TEST_HAS_NO_SPACESHIP_OPERATOR

struct one_way_eq
{
  bool operator==(one_way_eq const&) const = default;
  __host__ __device__ friend bool operator==(one_way_eq, explicit_operators);
  __host__ __device__ friend bool operator==(explicit_operators, one_way_eq) = delete;

  __host__ __device__ operator explicit_operators() const;
};

struct one_way_ne
{
  bool operator==(one_way_ne const&) const = default;
  __host__ __device__ friend bool operator==(one_way_ne, explicit_operators);
  __host__ __device__ friend bool operator!=(one_way_ne, explicit_operators) = delete;

  __host__ __device__ operator explicit_operators() const;
};
static_assert(requires(explicit_operators const x, one_way_ne const y) { x != y; });

struct explicit_bool
{
  __host__ __device__ explicit operator bool() const noexcept;
};

#  ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
struct totally_ordered_with_others
{
  auto operator<=>(totally_ordered_with_others const&) const = default;
};

struct no_lt_not_totally_ordered_with
{
  bool operator==(no_lt_not_totally_ordered_with const&) const  = default;
  auto operator<=>(no_lt_not_totally_ordered_with const&) const = default;
  __host__ __device__ operator totally_ordered_with_others() const noexcept;

  __host__ __device__ bool operator==(totally_ordered_with_others const&) const;
  __host__ __device__ auto operator<=>(totally_ordered_with_others const&) const;
  __host__ __device__ auto operator<(totally_ordered_with_others const&) const;
};

struct no_gt_not_totally_ordered_with
{
  bool operator==(no_gt_not_totally_ordered_with const&) const  = default;
  auto operator<=>(no_gt_not_totally_ordered_with const&) const = default;
  __host__ __device__ operator totally_ordered_with_others() const noexcept;

  __host__ __device__ bool operator==(totally_ordered_with_others const&) const;
  __host__ __device__ auto operator<=>(totally_ordered_with_others const&) const;
  __host__ __device__ auto operator>(totally_ordered_with_others const&) const;
};

struct no_le_not_totally_ordered_with
{
  bool operator==(no_le_not_totally_ordered_with const&) const  = default;
  auto operator<=>(no_le_not_totally_ordered_with const&) const = default;
  __host__ __device__ operator totally_ordered_with_others() const noexcept;

  __host__ __device__ bool operator==(totally_ordered_with_others const&) const;
  __host__ __device__ auto operator<=>(totally_ordered_with_others const&) const;
  __host__ __device__ auto operator<=(totally_ordered_with_others const&) const;
};

struct no_ge_not_totally_ordered_with
{
  bool operator==(no_ge_not_totally_ordered_with const&) const  = default;
  auto operator<=>(no_ge_not_totally_ordered_with const&) const = default;
  __host__ __device__ operator totally_ordered_with_others() const noexcept;

  __host__ __device__ bool operator==(totally_ordered_with_others const&) const;
  __host__ __device__ auto operator<=>(totally_ordered_with_others const&) const;
  __host__ __device__ auto operator>=(totally_ordered_with_others const&) const;
};

struct partial_ordering_totally_ordered_with
{
  auto operator<=>(partial_ordering_totally_ordered_with const&) const noexcept = default;
  __host__ __device__ std::partial_ordering operator<=>(totally_ordered_with_others const&) const noexcept;

  __host__ __device__ operator totally_ordered_with_others() const;
};

struct weak_ordering_totally_ordered_with
{
  auto operator<=>(weak_ordering_totally_ordered_with const&) const noexcept = default;
  __host__ __device__ std::weak_ordering operator<=>(totally_ordered_with_others const&) const noexcept;

  __host__ __device__ operator totally_ordered_with_others() const;
};

struct strong_ordering_totally_ordered_with
{
  auto operator<=>(strong_ordering_totally_ordered_with const&) const noexcept = default;
  __host__ __device__ std::strong_ordering operator<=>(totally_ordered_with_others const&) const noexcept;

  __host__ __device__ operator totally_ordered_with_others() const;
};

struct eq_returns_explicit_bool
{
  __host__ __device__ friend explicit_bool operator==(eq_returns_explicit_bool, eq_returns_explicit_bool);
  __host__ __device__ friend bool operator!=(eq_returns_explicit_bool, eq_returns_explicit_bool);
  __host__ __device__ friend bool operator<(eq_returns_explicit_bool, eq_returns_explicit_bool);
  __host__ __device__ friend bool operator>(eq_returns_explicit_bool, eq_returns_explicit_bool);
  __host__ __device__ friend bool operator<=(eq_returns_explicit_bool, eq_returns_explicit_bool);
  __host__ __device__ friend bool operator>=(eq_returns_explicit_bool, eq_returns_explicit_bool);

  __host__ __device__ operator totally_ordered_with_others() const;

  __host__ __device__ friend explicit_bool operator==(eq_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend explicit_bool operator==(totally_ordered_with_others, eq_returns_explicit_bool);
  __host__ __device__ friend bool operator!=(eq_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator!=(totally_ordered_with_others, eq_returns_explicit_bool);
  __host__ __device__ friend bool operator<(eq_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator<(totally_ordered_with_others, eq_returns_explicit_bool);
  __host__ __device__ friend bool operator>(eq_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator>(totally_ordered_with_others, eq_returns_explicit_bool);
  __host__ __device__ friend bool operator<=(eq_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator<=(totally_ordered_with_others, eq_returns_explicit_bool);
  __host__ __device__ friend bool operator>=(eq_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator>=(totally_ordered_with_others, eq_returns_explicit_bool);
};

struct ne_returns_explicit_bool
{
  __host__ __device__ friend bool operator==(ne_returns_explicit_bool, ne_returns_explicit_bool);
  __host__ __device__ friend explicit_bool operator!=(ne_returns_explicit_bool, ne_returns_explicit_bool);
  __host__ __device__ friend bool operator<(ne_returns_explicit_bool, ne_returns_explicit_bool);
  __host__ __device__ friend bool operator>(ne_returns_explicit_bool, ne_returns_explicit_bool);
  __host__ __device__ friend bool operator<=(ne_returns_explicit_bool, ne_returns_explicit_bool);
  __host__ __device__ friend bool operator>=(ne_returns_explicit_bool, ne_returns_explicit_bool);

  __host__ __device__ operator totally_ordered_with_others() const;

  __host__ __device__ friend bool operator==(ne_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend explicit_bool operator!=(ne_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend explicit_bool operator!=(totally_ordered_with_others, ne_returns_explicit_bool);
  __host__ __device__ friend bool operator<(ne_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator<(totally_ordered_with_others, ne_returns_explicit_bool);
  __host__ __device__ friend bool operator>(ne_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator>(totally_ordered_with_others, ne_returns_explicit_bool);
  __host__ __device__ friend bool operator<=(ne_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator<=(totally_ordered_with_others, ne_returns_explicit_bool);
  __host__ __device__ friend bool operator>=(ne_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator>=(totally_ordered_with_others, ne_returns_explicit_bool);
};

struct lt_returns_explicit_bool
{
  __host__ __device__ friend bool operator==(lt_returns_explicit_bool, lt_returns_explicit_bool);
  __host__ __device__ friend bool operator!=(lt_returns_explicit_bool, lt_returns_explicit_bool);
  __host__ __device__ friend explicit_bool operator<(lt_returns_explicit_bool, lt_returns_explicit_bool);
  __host__ __device__ friend bool operator>(lt_returns_explicit_bool, lt_returns_explicit_bool);
  __host__ __device__ friend bool operator<=(lt_returns_explicit_bool, lt_returns_explicit_bool);
  __host__ __device__ friend bool operator>=(lt_returns_explicit_bool, lt_returns_explicit_bool);

  __host__ __device__ operator totally_ordered_with_others() const;

  __host__ __device__ friend bool operator==(lt_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator!=(lt_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator!=(totally_ordered_with_others, lt_returns_explicit_bool);
  __host__ __device__ friend explicit_bool operator<(lt_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator<(totally_ordered_with_others, lt_returns_explicit_bool);
  __host__ __device__ friend bool operator>(lt_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator>(totally_ordered_with_others, lt_returns_explicit_bool);
  __host__ __device__ friend bool operator<=(lt_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator<=(totally_ordered_with_others, lt_returns_explicit_bool);
  __host__ __device__ friend bool operator>=(lt_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator>=(totally_ordered_with_others, lt_returns_explicit_bool);
};

struct gt_returns_explicit_bool
{
  __host__ __device__ friend bool operator==(gt_returns_explicit_bool, gt_returns_explicit_bool);
  __host__ __device__ friend bool operator!=(gt_returns_explicit_bool, gt_returns_explicit_bool);
  __host__ __device__ friend bool operator<(gt_returns_explicit_bool, gt_returns_explicit_bool);
  __host__ __device__ friend explicit_bool operator>(gt_returns_explicit_bool, gt_returns_explicit_bool);
  __host__ __device__ friend bool operator<=(gt_returns_explicit_bool, gt_returns_explicit_bool);
  __host__ __device__ friend bool operator>=(gt_returns_explicit_bool, gt_returns_explicit_bool);

  __host__ __device__ operator totally_ordered_with_others() const;

  __host__ __device__ friend bool operator==(gt_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator!=(gt_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator!=(totally_ordered_with_others, gt_returns_explicit_bool);
  __host__ __device__ friend bool operator<(gt_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator<(totally_ordered_with_others, gt_returns_explicit_bool);
  __host__ __device__ friend explicit_bool operator>(gt_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator>(totally_ordered_with_others, gt_returns_explicit_bool);
  __host__ __device__ friend bool operator<=(gt_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator<=(totally_ordered_with_others, gt_returns_explicit_bool);
  __host__ __device__ friend bool operator>=(gt_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator>=(totally_ordered_with_others, gt_returns_explicit_bool);
};

struct le_returns_explicit_bool
{
  __host__ __device__ friend bool operator==(le_returns_explicit_bool, le_returns_explicit_bool);
  __host__ __device__ friend bool operator!=(le_returns_explicit_bool, le_returns_explicit_bool);
  __host__ __device__ friend bool operator<(le_returns_explicit_bool, le_returns_explicit_bool);
  __host__ __device__ friend bool operator>(le_returns_explicit_bool, le_returns_explicit_bool);
  __host__ __device__ friend explicit_bool operator<=(le_returns_explicit_bool, le_returns_explicit_bool);
  __host__ __device__ friend bool operator>=(le_returns_explicit_bool, le_returns_explicit_bool);

  __host__ __device__ operator totally_ordered_with_others() const;

  __host__ __device__ friend bool operator==(le_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator!=(le_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator!=(totally_ordered_with_others, le_returns_explicit_bool);
  __host__ __device__ friend bool operator<(le_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator<(totally_ordered_with_others, le_returns_explicit_bool);
  __host__ __device__ friend bool operator>(le_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator>(totally_ordered_with_others, le_returns_explicit_bool);
  __host__ __device__ friend bool operator<=(le_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend explicit_bool operator<=(totally_ordered_with_others, le_returns_explicit_bool);
  __host__ __device__ friend bool operator>=(le_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator>=(totally_ordered_with_others, le_returns_explicit_bool);
};

struct ge_returns_explicit_bool
{
  __host__ __device__ friend bool operator==(ge_returns_explicit_bool, ge_returns_explicit_bool);
  __host__ __device__ friend bool operator!=(ge_returns_explicit_bool, ge_returns_explicit_bool);
  __host__ __device__ friend bool operator<(ge_returns_explicit_bool, ge_returns_explicit_bool);
  __host__ __device__ friend bool operator>(ge_returns_explicit_bool, ge_returns_explicit_bool);
  __host__ __device__ friend bool operator<=(ge_returns_explicit_bool, ge_returns_explicit_bool);
  __host__ __device__ friend explicit_bool operator>=(ge_returns_explicit_bool, ge_returns_explicit_bool);

  __host__ __device__ operator totally_ordered_with_others() const;

  __host__ __device__ friend bool operator==(ge_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator!=(ge_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator!=(totally_ordered_with_others, ge_returns_explicit_bool);
  __host__ __device__ friend bool operator<(ge_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator<(totally_ordered_with_others, ge_returns_explicit_bool);
  __host__ __device__ friend bool operator>(ge_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator>(totally_ordered_with_others, ge_returns_explicit_bool);
  __host__ __device__ friend bool operator<=(ge_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend bool operator<=(totally_ordered_with_others, ge_returns_explicit_bool);
  __host__ __device__ friend bool operator>=(ge_returns_explicit_bool, totally_ordered_with_others);
  __host__ __device__ friend explicit_bool operator>=(totally_ordered_with_others, ge_returns_explicit_bool);
};

struct returns_true_type
{
  __host__ __device__ friend cuda::std::true_type operator==(returns_true_type, returns_true_type);
  __host__ __device__ friend cuda::std::true_type operator!=(returns_true_type, returns_true_type);
  __host__ __device__ friend cuda::std::true_type operator<(returns_true_type, returns_true_type);
  __host__ __device__ friend cuda::std::true_type operator>(returns_true_type, returns_true_type);
  __host__ __device__ friend cuda::std::true_type operator<=(returns_true_type, returns_true_type);
  __host__ __device__ friend cuda::std::true_type operator>=(returns_true_type, returns_true_type);

  __host__ __device__ operator totally_ordered_with_others() const;

  __host__ __device__ friend cuda::std::true_type operator==(returns_true_type, totally_ordered_with_others);
  __host__ __device__ friend cuda::std::true_type operator==(totally_ordered_with_others, returns_true_type);
  __host__ __device__ friend cuda::std::true_type operator!=(returns_true_type, totally_ordered_with_others);
  __host__ __device__ friend cuda::std::true_type operator!=(totally_ordered_with_others, returns_true_type);
  __host__ __device__ friend cuda::std::true_type operator<(returns_true_type, totally_ordered_with_others);
  __host__ __device__ friend cuda::std::true_type operator<(totally_ordered_with_others, returns_true_type);
  __host__ __device__ friend cuda::std::true_type operator>(returns_true_type, totally_ordered_with_others);
  __host__ __device__ friend cuda::std::true_type operator>(totally_ordered_with_others, returns_true_type);
  __host__ __device__ friend cuda::std::true_type operator<=(returns_true_type, totally_ordered_with_others);
  __host__ __device__ friend cuda::std::true_type operator<=(totally_ordered_with_others, returns_true_type);
  __host__ __device__ friend cuda::std::true_type operator>=(returns_true_type, totally_ordered_with_others);
  __host__ __device__ friend cuda::std::true_type operator>=(totally_ordered_with_others, returns_true_type);
};

struct returns_int_ptr
{
  __host__ __device__ friend int* operator==(returns_int_ptr, returns_int_ptr);
  __host__ __device__ friend int* operator!=(returns_int_ptr, returns_int_ptr);
  __host__ __device__ friend int* operator<(returns_int_ptr, returns_int_ptr);
  __host__ __device__ friend int* operator>(returns_int_ptr, returns_int_ptr);
  __host__ __device__ friend int* operator<=(returns_int_ptr, returns_int_ptr);
  __host__ __device__ friend int* operator>=(returns_int_ptr, returns_int_ptr);

  __host__ __device__ operator totally_ordered_with_others() const;

  __host__ __device__ friend int* operator==(returns_int_ptr, totally_ordered_with_others);
  __host__ __device__ friend int* operator==(totally_ordered_with_others, returns_int_ptr);
  __host__ __device__ friend int* operator!=(returns_int_ptr, totally_ordered_with_others);
  __host__ __device__ friend int* operator!=(totally_ordered_with_others, returns_int_ptr);
  __host__ __device__ friend int* operator<(returns_int_ptr, totally_ordered_with_others);
  __host__ __device__ friend int* operator<(totally_ordered_with_others, returns_int_ptr);
  __host__ __device__ friend int* operator>(returns_int_ptr, totally_ordered_with_others);
  __host__ __device__ friend int* operator>(totally_ordered_with_others, returns_int_ptr);
  __host__ __device__ friend int* operator<=(returns_int_ptr, totally_ordered_with_others);
  __host__ __device__ friend int* operator<=(totally_ordered_with_others, returns_int_ptr);
  __host__ __device__ friend int* operator>=(returns_int_ptr, totally_ordered_with_others);
  __host__ __device__ friend int* operator>=(totally_ordered_with_others, returns_int_ptr);
};
#  endif // TEST_HAS_NO_SPACESHIP_OPERATOR
#endif // TEST_STD_VER > 2017

struct ForwardingTestObject
{
  __host__ __device__ constexpr bool operator<(ForwardingTestObject&&) &&
  {
    return true;
  }
  __host__ __device__ constexpr bool operator<(const ForwardingTestObject&) const&
  {
    return false;
  }

  __host__ __device__ constexpr bool operator==(ForwardingTestObject&&) &&
  {
    return true;
  }
  __host__ __device__ constexpr bool operator==(const ForwardingTestObject&) const&
  {
    return false;
  }

  __host__ __device__ constexpr bool operator!=(ForwardingTestObject&&) &&
  {
    return true;
  }
  __host__ __device__ constexpr bool operator!=(const ForwardingTestObject&) const&
  {
    return false;
  }

  __host__ __device__ constexpr bool operator<=(ForwardingTestObject&&) &&
  {
    return true;
  }
  __host__ __device__ constexpr bool operator<=(const ForwardingTestObject&) const&
  {
    return false;
  }

  __host__ __device__ constexpr bool operator>(ForwardingTestObject&&) &&
  {
    return true;
  }
  __host__ __device__ constexpr bool operator>(const ForwardingTestObject&) const&
  {
    return false;
  }

  __host__ __device__ constexpr bool operator>=(ForwardingTestObject&&) &&
  {
    return true;
  }
  __host__ __device__ constexpr bool operator>=(const ForwardingTestObject&) const&
  {
    return false;
  }
};

#endif // TEST_SUPPORT_COMPARE_TYPES_H
