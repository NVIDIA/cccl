//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_UTILITIES_EXPECTED_TYPES_H
#define TEST_STD_UTILITIES_EXPECTED_TYPES_H

#include <cuda/std/utility>

#include "test_macros.h"

template <bool copyMoveNoexcept, bool convertNoexcept = true>
struct TracedBase
{
  struct state
  {
    bool copyCtorCalled   = false;
    bool copyAssignCalled = false;
    bool moveCtorCalled   = false;
    bool moveAssignCalled = false;
    bool dtorCalled       = false;
  };

  state* state_      = nullptr;
  bool copiedFromInt = false;
  bool movedFromInt  = false;
  bool copiedFromTmp = false;
  bool movedFromTmp  = false;
  int data_;

  __host__ __device__ constexpr TracedBase(const int& ii) noexcept(convertNoexcept)
      : data_(ii)
  {
    copiedFromInt = true;
  }
  __host__ __device__ constexpr TracedBase(int&& ii) noexcept(convertNoexcept)
      : data_(ii)
  {
    movedFromInt = true;
  }
  __host__ __device__ constexpr TracedBase(state& s, int ii) noexcept
      : state_(&s)
      , data_(ii)
  {}
  __host__ __device__ constexpr TracedBase(const TracedBase& other) noexcept(copyMoveNoexcept)
      : state_(other.state_)
      , data_(other.data_)
  {
    if (state_)
    {
      state_->copyCtorCalled = true;
    }
    else
    {
      copiedFromTmp = true;
    }
  }
  __host__ __device__ constexpr TracedBase(TracedBase&& other) noexcept(copyMoveNoexcept)
      : state_(other.state_)
      , data_(other.data_)
  {
    if (state_)
    {
      state_->moveCtorCalled = true;
    }
    else
    {
      movedFromTmp = true;
    }
  }
  __host__ __device__ constexpr TracedBase& operator=(const TracedBase& other) noexcept(copyMoveNoexcept)
  {
    data_                    = other.data_;
    state_->copyAssignCalled = true;
    return *this;
  }
  __host__ __device__ constexpr TracedBase& operator=(TracedBase&& other) noexcept(copyMoveNoexcept)
  {
    data_                    = other.data_;
    state_->moveAssignCalled = true;
    return *this;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX20 ~TracedBase()
  {
    if (state_)
    {
      state_->dtorCalled = true;
    }
  }
};

using Traced         = TracedBase<false>;
using TracedNoexcept = TracedBase<true>;

using MoveThrowConvNoexcept = TracedBase<false, true>;
using MoveNoexceptConvThrow = TracedBase<true, false>;
using BothMayThrow          = TracedBase<false, false>;
using BothNoexcept          = TracedBase<true, true>;

struct ADLSwap
{
  int i;
  bool adlSwapCalled = false;
  __host__ __device__ constexpr ADLSwap(int ii)
      : i(ii)
  {}
  __host__ __device__ constexpr friend void swap(ADLSwap& x, ADLSwap& y)
  {
    cuda::std::swap(x.i, y.i);
    x.adlSwapCalled = true;
    y.adlSwapCalled = true;
  }
};

template <bool Noexcept>
struct TrackedMove
{
  int i;
  int numberOfMoves = 0;
  bool swapCalled   = false;

  __host__ __device__ constexpr TrackedMove(int ii)
      : i(ii)
  {}
  __host__ __device__ constexpr TrackedMove(TrackedMove&& other) noexcept(Noexcept)
      : i(other.i)
      , numberOfMoves(other.numberOfMoves)
      , swapCalled(other.swapCalled)
  {
    ++numberOfMoves;
  }

  __host__ __device__ constexpr friend void swap(TrackedMove& x, TrackedMove& y)
  {
    cuda::std::swap(x.i, y.i);
    cuda::std::swap(x.numberOfMoves, y.numberOfMoves);
    x.swapCalled = true;
    y.swapCalled = true;
  }
};

#if TEST_HAS_EXCEPTIONS()
struct Except
{};

struct ThrowOnCopyConstruct
{
  ThrowOnCopyConstruct() = default;
  ThrowOnCopyConstruct(const ThrowOnCopyConstruct&)
  {
    throw Except{};
  }
  ThrowOnCopyConstruct& operator=(const ThrowOnCopyConstruct&) = default;
};

struct ThrowOnMoveConstruct
{
  ThrowOnMoveConstruct() = default;
  ThrowOnMoveConstruct(ThrowOnMoveConstruct&&)
  {
    throw Except{};
  }
  ThrowOnMoveConstruct& operator=(ThrowOnMoveConstruct&&) = default;
};

struct ThrowOnConvert
{
  ThrowOnConvert() = default;
  ThrowOnConvert(const int&)
  {
    throw Except{};
  }
  ThrowOnConvert(int&&)
  {
    throw Except{};
  }
  ThrowOnConvert(const ThrowOnConvert&) noexcept(false) {}
  ThrowOnConvert& operator=(const ThrowOnConvert&) = default;
  ThrowOnConvert(ThrowOnConvert&&) noexcept(false) {}
  ThrowOnConvert& operator=(ThrowOnConvert&&) = default;
};

struct ThrowOnMove
{
  bool* destroyed = nullptr;
  ThrowOnMove()   = default;
  ThrowOnMove(bool& d)
      : destroyed(&d)
  {}
  ThrowOnMove(ThrowOnMove&&)
  {
    throw Except{};
  };
  ThrowOnMove& operator=(ThrowOnMove&&) = default;
  ~ThrowOnMove()
  {
    if (destroyed)
    {
      *destroyed = true;
    }
  }
};

#endif // TEST_HAS_EXCEPTIONS()

struct TestError
{
  __host__ __device__ constexpr TestError(const int err) noexcept
      : err_(err)
  {}

  __host__ __device__ friend constexpr bool operator==(const TestError& lhs, const TestError& rhs) noexcept
  {
    return lhs.err_ == rhs.err_;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator!=(const TestError& lhs, const TestError& rhs) noexcept
  {
    return lhs.err_ != rhs.err_;
  }
#endif
  int err_ = 1;
};

#endif // TEST_STD_UTILITIES_EXPECTED_TYPES_H
