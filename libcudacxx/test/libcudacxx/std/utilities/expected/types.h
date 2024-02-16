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
struct TracedBase {
  struct state {
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

  TEST_HOST_DEVICE constexpr TracedBase(const int& ii) noexcept(convertNoexcept) : data_(ii) { copiedFromInt = true; }
  TEST_HOST_DEVICE constexpr TracedBase(int&& ii) noexcept(convertNoexcept) : data_(ii) { movedFromInt = true; }
  TEST_HOST_DEVICE constexpr TracedBase(state& s, int ii) noexcept : state_(&s), data_(ii) {}
  TEST_HOST_DEVICE constexpr TracedBase(const TracedBase& other) noexcept(copyMoveNoexcept) : state_(other.state_), data_(other.data_) {
    if (state_) {
      state_->copyCtorCalled = true;
    } else {
      copiedFromTmp = true;
    }
  }
  TEST_HOST_DEVICE constexpr TracedBase(TracedBase&& other) noexcept(copyMoveNoexcept) : state_(other.state_), data_(other.data_) {
    if (state_) {
      state_->moveCtorCalled = true;
    } else {
      movedFromTmp = true;
    }
  }
  TEST_HOST_DEVICE constexpr TracedBase& operator=(const TracedBase& other) noexcept(copyMoveNoexcept) {
    data_                    = other.data_;
    state_->copyAssignCalled = true;
    return *this;
  }
  TEST_HOST_DEVICE constexpr TracedBase& operator=(TracedBase&& other) noexcept(copyMoveNoexcept) {
    data_                    = other.data_;
    state_->moveAssignCalled = true;
    return *this;
  }
  TEST_HOST_DEVICE TEST_CONSTEXPR_CXX20 ~TracedBase() {
    if (state_) {
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

struct ADLSwap {
  int i;
  bool adlSwapCalled = false;
  TEST_HOST_DEVICE constexpr ADLSwap(int ii) : i(ii) {}
  TEST_HOST_DEVICE constexpr friend void swap(ADLSwap& x, ADLSwap& y) {
    cuda::std::swap(x.i, y.i);
    x.adlSwapCalled = true;
    y.adlSwapCalled = true;
  }
};

template <bool Noexcept>
struct TrackedMove {
  int i;
  int numberOfMoves = 0;
  bool swapCalled   = false;

  TEST_HOST_DEVICE constexpr TrackedMove(int ii) : i(ii) {}
  TEST_HOST_DEVICE constexpr TrackedMove(TrackedMove&& other) noexcept(Noexcept)
      : i(other.i), numberOfMoves(other.numberOfMoves), swapCalled(other.swapCalled) {
    ++numberOfMoves;
  }

  TEST_HOST_DEVICE constexpr friend void swap(TrackedMove& x, TrackedMove& y) {
    cuda::std::swap(x.i, y.i);
    cuda::std::swap(x.numberOfMoves, y.numberOfMoves);
    x.swapCalled = true;
    y.swapCalled = true;
  }
};

#ifndef TEST_HAS_NO_EXCEPTIONS
struct Except {};

struct ThrowOnCopyConstruct {
  ThrowOnCopyConstruct() = default;
  TEST_HOST_DEVICE ThrowOnCopyConstruct(const ThrowOnCopyConstruct&) { throw Except{}; }
  TEST_HOST_DEVICE ThrowOnCopyConstruct& operator=(const ThrowOnCopyConstruct&) = default;
};

struct ThrowOnMoveConstruct {
  ThrowOnMoveConstruct() = default;
  TEST_HOST_DEVICE ThrowOnMoveConstruct(ThrowOnMoveConstruct&&) { throw Except{}; }
  TEST_HOST_DEVICE ThrowOnMoveConstruct& operator=(ThrowOnMoveConstruct&&) = default;
};

struct ThrowOnConvert {
  ThrowOnConvert() = default;
  TEST_HOST_DEVICE ThrowOnConvert(const int&) { throw Except{}; }
  TEST_HOST_DEVICE ThrowOnConvert(int&&) { throw Except{}; }
  TEST_HOST_DEVICE ThrowOnConvert(const ThrowOnConvert&) noexcept(false) {}
  ThrowOnConvert& operator=(const ThrowOnConvert&) = default;
  TEST_HOST_DEVICE ThrowOnConvert(ThrowOnConvert&&) noexcept(false) {}
  ThrowOnConvert& operator=(ThrowOnConvert&&) = default;
};

struct ThrowOnMove {
  bool* destroyed = nullptr;
  ThrowOnMove()   = default;
  TEST_HOST_DEVICE ThrowOnMove(bool& d) : destroyed(&d) {}
  TEST_HOST_DEVICE ThrowOnMove(ThrowOnMove&&) { throw Except{}; };
  TEST_HOST_DEVICE ThrowOnMove& operator=(ThrowOnMove&&) = default;
  TEST_HOST_DEVICE ~ThrowOnMove() {
    if (destroyed) {
      *destroyed = true;
    }
  }
};

#endif // TEST_HAS_NO_EXCEPTIONS

struct TestError {
    TEST_HOST_DEVICE
    constexpr TestError(const int err) noexcept : err_(err) {}

    TEST_HOST_DEVICE
    friend constexpr bool operator==(const TestError& lhs, const TestError& rhs) noexcept {
        return lhs.err_ == rhs.err_;
    }
#if TEST_STD_VER < 2020
    TEST_HOST_DEVICE
    friend constexpr bool operator!=(const TestError& lhs, const TestError& rhs) noexcept {
        return lhs.err_ != rhs.err_;
    }
#endif
    int err_ = 1;
};

#endif // TEST_STD_UTILITIES_EXPECTED_TYPES_H
