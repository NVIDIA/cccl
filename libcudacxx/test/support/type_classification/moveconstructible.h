//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef TEST_SUPPORT_TYPE_CLASSIFICATION_MOVECONSTRUCTIBLE_H
#define TEST_SUPPORT_TYPE_CLASSIFICATION_MOVECONSTRUCTIBLE_H

#include "test_macros.h"

struct HasDefaultOps {};

struct CustomMoveCtor {
  TEST_HOST_DEVICE CustomMoveCtor(CustomMoveCtor&&) noexcept;
};

struct MoveOnly {
  MoveOnly(MoveOnly&&) noexcept = default;
  MoveOnly& operator=(MoveOnly&&) noexcept = default;
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly& operator=(const MoveOnly&) = default;
};

struct CustomMoveAssign {
  TEST_HOST_DEVICE CustomMoveAssign(CustomMoveAssign&&) noexcept;
  TEST_HOST_DEVICE CustomMoveAssign& operator=(CustomMoveAssign&&) noexcept;
};

struct DeletedMoveCtor {
  DeletedMoveCtor(DeletedMoveCtor&&) = delete;
  DeletedMoveCtor& operator=(DeletedMoveCtor&&) = default;
};

struct ImplicitlyDeletedMoveCtor {
  DeletedMoveCtor X;
};

struct DeletedMoveAssign {
  DeletedMoveAssign& operator=(DeletedMoveAssign&&) = delete;
};

struct ImplicitlyDeletedMoveAssign {
  DeletedMoveAssign X;
};

class MemberLvalueReference {
public:
  TEST_HOST_DEVICE MemberLvalueReference(int&);

private:
  int& X;
};

class MemberRvalueReference {
public:
  TEST_HOST_DEVICE MemberRvalueReference(int&&);

private:
  int&& X;
};

struct NonMovable {
  NonMovable() = default;
  NonMovable(NonMovable&&) = delete;
  NonMovable& operator=(NonMovable&&) = delete;
};

struct DerivedFromNonMovable : NonMovable {};

struct HasANonMovable {
  NonMovable X;
};

#endif // TEST_SUPPORT_TYPE_CLASSIFICATION_MOVECONSTRUCTIBLE_H
