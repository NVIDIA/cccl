//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_TEST_SUPPORT_BOOLEAN_TESTABLE_H
#define LIBCXX_TEST_SUPPORT_BOOLEAN_TESTABLE_H

#include "test_macros.h"

#if TEST_STD_VER >= 2017

class BooleanTestable
{
public:
  __host__ __device__ constexpr operator bool() const
  {
    return value_;
  }

  __host__ __device__ friend constexpr BooleanTestable operator==(const BooleanTestable& lhs, const BooleanTestable& rhs)
  {
    return lhs.value_ == rhs.value_;
  }

  __host__ __device__ friend constexpr BooleanTestable operator!=(const BooleanTestable& lhs, const BooleanTestable& rhs)
  {
    return !(lhs == rhs);
  }

  __host__ __device__ constexpr BooleanTestable operator!()
  {
    return BooleanTestable{!value_};
  }

  // this class should behave like a bool, so the constructor shouldn't be explicit
  __host__ __device__ constexpr BooleanTestable(bool value)
      : value_{value}
  {}
  constexpr BooleanTestable(const BooleanTestable&) = delete;
  constexpr BooleanTestable(BooleanTestable&&)      = delete;

private:
  bool value_;
};

template <class T>
class StrictComparable
{
public:
  // this shouldn't be explicit to make it easier to initlaize inside arrays (which it almost always is)
  __host__ __device__ constexpr StrictComparable(T value)
      : value_{value}
  {}

  __host__ __device__ friend constexpr BooleanTestable
  operator==(const StrictComparable& lhs, const StrictComparable& rhs)
  {
    return (lhs.value_ == rhs.value_);
  }

  __host__ __device__ friend constexpr BooleanTestable
  operator!=(const StrictComparable& lhs, const StrictComparable& rhs)
  {
    return !(lhs == rhs);
  }

private:
  T value_;
};

#endif // TEST_STD_VER >= 2017

#endif // LIBCXX_TEST_SUPPORT_BOOLEAN_TESTABLE_H
