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
// concept copyable = see below;

#include "type_classification/copyable.h"

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::copyable;

static_assert(copyable<int>, "");
static_assert(copyable<int volatile>, "");
static_assert(copyable<int*>, "");
static_assert(copyable<int const*>, "");
static_assert(copyable<int volatile*>, "");
static_assert(copyable<int volatile const*>, "");
static_assert(copyable<int (*)()>, "");

struct S
{};
static_assert(copyable<S>, "");
static_assert(copyable<int S::*>, "");
static_assert(copyable<int (S::*)()>, "");
static_assert(copyable<int (S::*)() noexcept>, "");
static_assert(copyable<int (S::*)() &>, "");
static_assert(copyable<int (S::*)() & noexcept>, "");
static_assert(copyable<int (S::*)() &&>, "");
static_assert(copyable < int(S::*)() && noexcept >, "");
static_assert(copyable<int (S::*)() const>, "");
static_assert(copyable<int (S::*)() const noexcept>, "");
static_assert(copyable<int (S::*)() const&>, "");
static_assert(copyable<int (S::*)() const & noexcept>, "");
static_assert(copyable<int (S::*)() const&&>, "");
static_assert(copyable < int(S::*)() const&& noexcept >, "");
static_assert(copyable<int (S::*)() volatile>, "");
static_assert(copyable<int (S::*)() volatile noexcept>, "");
static_assert(copyable<int (S::*)() volatile&>, "");
static_assert(copyable<int (S::*)() volatile & noexcept>, "");
static_assert(copyable<int (S::*)() volatile&&>, "");
static_assert(copyable < int(S::*)() volatile && noexcept >, "");
static_assert(copyable<int (S::*)() const volatile>, "");
static_assert(copyable<int (S::*)() const volatile noexcept>, "");
static_assert(copyable<int (S::*)() const volatile&>, "");
static_assert(copyable<int (S::*)() const volatile & noexcept>, "");
static_assert(copyable<int (S::*)() const volatile&&>, "");
static_assert(copyable < int(S::*)() const volatile&& noexcept >, "");

static_assert(copyable<has_volatile_member>, "");
static_assert(copyable<has_array_member>, "");

// Not objects
static_assert(!copyable<void>, "");
static_assert(!copyable<int&>, "");
static_assert(!copyable<int const&>, "");
static_assert(!copyable<int volatile&>, "");
static_assert(!copyable<int const volatile&>, "");
static_assert(!copyable<int&&>, "");
static_assert(!copyable<int const&&>, "");
static_assert(!copyable<int volatile&&>, "");
static_assert(!copyable<int const volatile&&>, "");
static_assert(!copyable<int()>, "");
static_assert(!copyable<int (&)()>, "");
static_assert(!copyable<int[5]>, "");

// Not assignable
static_assert(!copyable<int const>, "");
static_assert(!copyable<int const volatile>, "");
static_assert(copyable<const_copy_assignment const>, "");
static_assert(!copyable<volatile_copy_assignment volatile>, "");
static_assert(copyable<cv_copy_assignment const volatile>, "");

static_assert(!copyable<no_copy_constructor>, "");
static_assert(!copyable<no_copy_assignment>, "");

#if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017 // MSVC chokes on multiple definitions of SMF
static_assert(cuda::std::is_copy_assignable_v<no_copy_assignment_mutable>, "");
static_assert(!copyable<no_copy_assignment_mutable>, "");
#endif // !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017
static_assert(!copyable<derived_from_noncopyable>, "");
static_assert(!copyable<has_noncopyable>, "");
static_assert(!copyable<has_const_member>, "");
static_assert(!copyable<has_cv_member>, "");
static_assert(!copyable<has_lvalue_reference_member>, "");
static_assert(!copyable<has_rvalue_reference_member>, "");
static_assert(!copyable<has_function_ref_member>, "");

static_assert(
  !cuda::std::assignable_from<deleted_assignment_from_const_rvalue&, deleted_assignment_from_const_rvalue const>, "");
static_assert(!copyable<deleted_assignment_from_const_rvalue>, "");

int main(int, char**)
{
  return 0;
}
