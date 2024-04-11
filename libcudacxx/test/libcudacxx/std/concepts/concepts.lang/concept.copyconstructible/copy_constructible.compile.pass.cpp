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
// concept copy_constructible;

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "test_macros.h"
#include "type_classification/moveconstructible.h"

using cuda::std::copy_constructible;

// Tests in this namespace are shared with moveconstructible.pass.cpp
// There are some interesting differences, so it's best if they're tested here
// too.
namespace MoveConstructibleTests
{
static_assert(copy_constructible<int>, "");
static_assert(copy_constructible<int*>, "");
static_assert(copy_constructible<int&>, "");
static_assert(copy_constructible<const int>, "");
static_assert(copy_constructible<const int&>, "");
static_assert(copy_constructible<volatile int>, "");
static_assert(copy_constructible<volatile int&>, "");
static_assert(copy_constructible<int (*)()>, "");
static_assert(copy_constructible<int (&)()>, "");
static_assert(copy_constructible<HasDefaultOps>, "");
static_assert(copy_constructible<const CustomMoveCtor&>, "");
static_assert(copy_constructible<volatile CustomMoveCtor&>, "");
static_assert(copy_constructible<const CustomMoveAssign&>, "");
static_assert(copy_constructible<volatile CustomMoveAssign&>, "");
static_assert(copy_constructible<int HasDefaultOps::*>, "");
static_assert(copy_constructible<void (HasDefaultOps::*)(int)>, "");
static_assert(copy_constructible<MemberLvalueReference>, "");

static_assert(!copy_constructible<void>, "");
static_assert(!copy_constructible<CustomMoveAssign>, "");
static_assert(!copy_constructible<const CustomMoveCtor>, "");
static_assert(!copy_constructible<volatile CustomMoveCtor>, "");
static_assert(!copy_constructible<const CustomMoveAssign>, "");
static_assert(!copy_constructible<volatile CustomMoveAssign>, "");
static_assert(!copy_constructible<int[10]>, "");
static_assert(!copy_constructible<DeletedMoveCtor>, "");
static_assert(!copy_constructible<ImplicitlyDeletedMoveCtor>, "");
static_assert(!copy_constructible<DeletedMoveAssign>, "");
static_assert(!copy_constructible<ImplicitlyDeletedMoveAssign>, "");

static_assert(copy_constructible<DeletedMoveCtor&>, "");
static_assert(copy_constructible<const DeletedMoveCtor&>, "");
static_assert(copy_constructible<ImplicitlyDeletedMoveCtor&>, "");
static_assert(copy_constructible<const ImplicitlyDeletedMoveCtor&>, "");
static_assert(copy_constructible<DeletedMoveAssign&>, "");
static_assert(copy_constructible<const DeletedMoveAssign&>, "");
static_assert(copy_constructible<ImplicitlyDeletedMoveAssign&>, "");
static_assert(copy_constructible<const ImplicitlyDeletedMoveAssign&>, "");

// different to moveconstructible.pass.cpp
static_assert(!copy_constructible<int&&>, "");
static_assert(!copy_constructible<const int&&>, "");
static_assert(!copy_constructible<volatile int&&>, "");
static_assert(!copy_constructible<CustomMoveCtor>, "");
static_assert(!copy_constructible<MoveOnly>, "");
static_assert(!copy_constructible<const CustomMoveCtor&&>, "");
static_assert(!copy_constructible<volatile CustomMoveCtor&&>, "");
static_assert(!copy_constructible<const CustomMoveAssign&&>, "");
static_assert(!copy_constructible<volatile CustomMoveAssign&&>, "");
static_assert(!copy_constructible<DeletedMoveCtor&&>, "");
static_assert(!copy_constructible<const DeletedMoveCtor&&>, "");
static_assert(!copy_constructible<ImplicitlyDeletedMoveCtor&&>, "");
static_assert(!copy_constructible<const ImplicitlyDeletedMoveCtor&&>, "");
static_assert(!copy_constructible<DeletedMoveAssign&&>, "");
static_assert(!copy_constructible<const DeletedMoveAssign&&>, "");
static_assert(!copy_constructible<ImplicitlyDeletedMoveAssign&&>, "");
static_assert(!copy_constructible<const ImplicitlyDeletedMoveAssign&&>, "");
static_assert(!copy_constructible<MemberRvalueReference>, "");
} // namespace MoveConstructibleTests

namespace CopyConstructibleTests
{
struct CopyCtorUserDefined
{
  CopyCtorUserDefined(CopyCtorUserDefined&&) noexcept = default;
  __host__ __device__ CopyCtorUserDefined(const CopyCtorUserDefined&);
};
static_assert(copy_constructible<CopyCtorUserDefined>, "");

struct CopyAssignUserDefined
{
  CopyAssignUserDefined& operator=(CopyAssignUserDefined&&) noexcept = default;
  __host__ __device__ CopyAssignUserDefined& operator=(const CopyAssignUserDefined&);
};
static_assert(!copy_constructible<CopyAssignUserDefined>, "");

struct CopyCtorAndAssignUserDefined
{
  CopyCtorAndAssignUserDefined(CopyCtorAndAssignUserDefined&&) noexcept = default;
  __host__ __device__ CopyCtorAndAssignUserDefined(const CopyCtorAndAssignUserDefined&);
  CopyCtorAndAssignUserDefined& operator=(CopyCtorAndAssignUserDefined&&) noexcept = default;
  __host__ __device__ CopyCtorAndAssignUserDefined& operator=(const CopyCtorAndAssignUserDefined&);
};
static_assert(copy_constructible<CopyCtorAndAssignUserDefined>, "");

struct CopyCtorDeleted
{
  CopyCtorDeleted(CopyCtorDeleted&&) noexcept = default;
  CopyCtorDeleted(const CopyCtorDeleted&)     = delete;
};
static_assert(!copy_constructible<CopyCtorDeleted>, "");

struct CopyAssignDeleted
{
  CopyAssignDeleted(CopyAssignDeleted&&) noexcept = default;
  CopyAssignDeleted(const CopyAssignDeleted&)     = delete;
};
static_assert(!copy_constructible<CopyAssignDeleted>, "");

struct CopyCtorHasMutableRef
{
  CopyCtorHasMutableRef(CopyCtorHasMutableRef&&) noexcept = default;
  CopyCtorHasMutableRef(CopyCtorHasMutableRef&)           = default;
};
static_assert(!copy_constructible<CopyCtorHasMutableRef>, "");

#if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017 // MSVC chokes on the deleted copy constructor
struct CopyCtorProhibitsMutableRef
{
  CopyCtorProhibitsMutableRef(CopyCtorProhibitsMutableRef&&) noexcept = default;
  CopyCtorProhibitsMutableRef(const CopyCtorProhibitsMutableRef&)     = default;
  CopyCtorProhibitsMutableRef(CopyCtorProhibitsMutableRef&)           = delete;
};
static_assert(!copy_constructible<CopyCtorProhibitsMutableRef>, "");
#endif // !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017

struct CopyAssignHasMutableRef
{
  CopyAssignHasMutableRef& operator=(CopyAssignHasMutableRef&&) noexcept = default;
  CopyAssignHasMutableRef& operator=(CopyAssignHasMutableRef&)           = default;
};
static_assert(!copy_constructible<CopyAssignHasMutableRef>, "");

#if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017 // MSVC chokes on the deleted copy assignment
struct CopyAssignProhibitsMutableRef
{
  CopyAssignProhibitsMutableRef& operator=(CopyAssignProhibitsMutableRef&&) noexcept = default;
  CopyAssignProhibitsMutableRef& operator=(const CopyAssignProhibitsMutableRef&)     = default;
  CopyAssignProhibitsMutableRef& operator=(CopyAssignProhibitsMutableRef&)           = delete;
};
static_assert(!copy_constructible<CopyAssignProhibitsMutableRef>, "");
#endif // !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017

struct CopyCtorOnly
{
  CopyCtorOnly(CopyCtorOnly&&) noexcept = delete;
  CopyCtorOnly(const CopyCtorOnly&)     = default;
};
static_assert(!copy_constructible<CopyCtorOnly>, "");

struct CopyAssignOnly
{
  CopyAssignOnly& operator=(CopyAssignOnly&&) noexcept = delete;
  CopyAssignOnly& operator=(const CopyAssignOnly&)     = default;
};
static_assert(!copy_constructible<CopyAssignOnly>, "");

struct CopyOnly
{
  CopyOnly(CopyOnly&&) noexcept = delete;
  CopyOnly(const CopyOnly&)     = default;

  CopyOnly& operator=(CopyOnly&&) noexcept = delete;
  CopyOnly& operator=(const CopyOnly&)     = default;
};
static_assert(!copy_constructible<CopyOnly>, "");

struct ExplicitlyCopyable
{
  ExplicitlyCopyable(ExplicitlyCopyable&&) = default;
  __host__ __device__ explicit ExplicitlyCopyable(const ExplicitlyCopyable&);
};
static_assert(!copy_constructible<ExplicitlyCopyable>, "");
} // namespace CopyConstructibleTests

int main(int, char**)
{
  return 0;
}
