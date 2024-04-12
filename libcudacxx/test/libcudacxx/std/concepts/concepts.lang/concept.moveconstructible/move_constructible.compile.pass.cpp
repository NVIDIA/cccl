//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class T>
// concept move_constructible;

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "test_macros.h"
#include "type_classification/moveconstructible.h"

using cuda::std::move_constructible;

static_assert(move_constructible<int>, "");
static_assert(move_constructible<int*>, "");
static_assert(move_constructible<int&>, "");
static_assert(move_constructible<int&&>, "");
static_assert(move_constructible<const int>, "");
static_assert(move_constructible<const int&>, "");
static_assert(move_constructible<const int&&>, "");
static_assert(move_constructible<volatile int>, "");
static_assert(move_constructible<volatile int&>, "");
static_assert(move_constructible<volatile int&&>, "");
static_assert(move_constructible<int (*)()>, "");
static_assert(move_constructible<int (&)()>, "");
static_assert(move_constructible<HasDefaultOps>, "");
static_assert(move_constructible<CustomMoveCtor>, "");
static_assert(move_constructible<MoveOnly>, "");
static_assert(move_constructible<const CustomMoveCtor&>, "");
static_assert(move_constructible<volatile CustomMoveCtor&>, "");
static_assert(move_constructible<const CustomMoveCtor&&>, "");
static_assert(move_constructible<volatile CustomMoveCtor&&>, "");
static_assert(move_constructible<CustomMoveAssign>, "");
static_assert(move_constructible<const CustomMoveAssign&>, "");
static_assert(move_constructible<volatile CustomMoveAssign&>, "");
static_assert(move_constructible<const CustomMoveAssign&&>, "");
static_assert(move_constructible<volatile CustomMoveAssign&&>, "");
static_assert(move_constructible<int HasDefaultOps::*>, "");
static_assert(move_constructible<void (HasDefaultOps::*)(int)>, "");
static_assert(move_constructible<MemberLvalueReference>, "");
static_assert(move_constructible<MemberRvalueReference>, "");

static_assert(!move_constructible<void>, "");
static_assert(!move_constructible<const CustomMoveCtor>, "");
static_assert(!move_constructible<volatile CustomMoveCtor>, "");
static_assert(!move_constructible<const CustomMoveAssign>, "");
static_assert(!move_constructible<volatile CustomMoveAssign>, "");
static_assert(!move_constructible<int[10]>, "");
static_assert(!move_constructible<DeletedMoveCtor>, "");
static_assert(!move_constructible<ImplicitlyDeletedMoveCtor>, "");
static_assert(!move_constructible<DeletedMoveAssign>, "");
static_assert(!move_constructible<ImplicitlyDeletedMoveAssign>, "");

static_assert(move_constructible<DeletedMoveCtor&>, "");
static_assert(move_constructible<DeletedMoveCtor&&>, "");
static_assert(move_constructible<const DeletedMoveCtor&>, "");
static_assert(move_constructible<const DeletedMoveCtor&&>, "");
static_assert(move_constructible<ImplicitlyDeletedMoveCtor&>, "");
static_assert(move_constructible<ImplicitlyDeletedMoveCtor&&>, "");
static_assert(move_constructible<const ImplicitlyDeletedMoveCtor&>, "");
static_assert(move_constructible<const ImplicitlyDeletedMoveCtor&&>, "");
static_assert(move_constructible<DeletedMoveAssign&>, "");
static_assert(move_constructible<DeletedMoveAssign&&>, "");
static_assert(move_constructible<const DeletedMoveAssign&>, "");
static_assert(move_constructible<const DeletedMoveAssign&&>, "");
static_assert(move_constructible<ImplicitlyDeletedMoveAssign&>, "");
static_assert(move_constructible<ImplicitlyDeletedMoveAssign&&>, "");
static_assert(move_constructible<const ImplicitlyDeletedMoveAssign&>, "");
static_assert(move_constructible<const ImplicitlyDeletedMoveAssign&&>, "");

static_assert(!move_constructible<NonMovable>, "");
static_assert(!move_constructible<DerivedFromNonMovable>, "");
static_assert(!move_constructible<HasANonMovable>, "");

int main(int, char**)
{
  return 0;
}
