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
// concept semiregular = see below;

#include "type_classification/semiregular.h"

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::semiregular;

static_assert(semiregular<int>, "");
static_assert(semiregular<int volatile>, "");
static_assert(semiregular<int*>, "");
static_assert(semiregular<int const*>, "");
static_assert(semiregular<int volatile*>, "");
static_assert(semiregular<int volatile const*>, "");
static_assert(semiregular<int (*)()>, "");

struct S
{};
static_assert(semiregular<S>, "");
static_assert(semiregular<int S::*>, "");
static_assert(semiregular<int (S::*)()>, "");
static_assert(semiregular<int (S::*)() noexcept>, "");
static_assert(semiregular<int (S::*)() &>, "");
static_assert(semiregular<int (S::*)() & noexcept>, "");
static_assert(semiregular<int (S::*)() &&>, "");
static_assert(semiregular < int(S::*)() && noexcept >, "");
static_assert(semiregular<int (S::*)() const>, "");
static_assert(semiregular<int (S::*)() const noexcept>, "");
static_assert(semiregular<int (S::*)() const&>, "");
static_assert(semiregular<int (S::*)() const & noexcept>, "");
static_assert(semiregular<int (S::*)() const&&>, "");
static_assert(semiregular < int(S::*)() const&& noexcept >, "");
static_assert(semiregular<int (S::*)() volatile>, "");
static_assert(semiregular<int (S::*)() volatile noexcept>, "");
static_assert(semiregular<int (S::*)() volatile&>, "");
static_assert(semiregular<int (S::*)() volatile & noexcept>, "");
static_assert(semiregular<int (S::*)() volatile&&>, "");
static_assert(semiregular < int(S::*)() volatile && noexcept >, "");
static_assert(semiregular<int (S::*)() const volatile>, "");
static_assert(semiregular<int (S::*)() const volatile noexcept>, "");
static_assert(semiregular<int (S::*)() const volatile&>, "");
static_assert(semiregular<int (S::*)() const volatile & noexcept>, "");
static_assert(semiregular<int (S::*)() const volatile&&>, "");
static_assert(semiregular < int(S::*)() const volatile&& noexcept >, "");

static_assert(semiregular<has_volatile_member>, "");
static_assert(semiregular<has_array_member>, "");

// Not objects
static_assert(!semiregular<void>, "");
static_assert(!semiregular<int&>, "");
static_assert(!semiregular<int const&>, "");
static_assert(!semiregular<int volatile&>, "");
static_assert(!semiregular<int const volatile&>, "");
static_assert(!semiregular<int&&>, "");
static_assert(!semiregular<int const&&>, "");
static_assert(!semiregular<int volatile&&>, "");
static_assert(!semiregular<int const volatile&&>, "");
static_assert(!semiregular<int()>, "");
static_assert(!semiregular<int (&)()>, "");
static_assert(!semiregular<int[5]>, "");

// Not copyable
static_assert(!semiregular<int const>, "");
static_assert(!semiregular<int const volatile>, "");
static_assert(semiregular<const_copy_assignment const>, "");
static_assert(!semiregular<volatile_copy_assignment volatile>, "");
static_assert(semiregular<cv_copy_assignment const volatile>, "");
static_assert(!semiregular<no_copy_constructor>, "");
static_assert(!semiregular<no_copy_assignment>, "");
#if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017 // MSVC chokes on multiple definitions of SMF
static_assert(!semiregular<no_copy_assignment_mutable>, "");
#endif // !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017
static_assert(!semiregular<derived_from_noncopyable>, "");
static_assert(!semiregular<has_noncopyable>, "");
static_assert(!semiregular<has_const_member>, "");
static_assert(!semiregular<has_cv_member>, "");
static_assert(!semiregular<has_lvalue_reference_member>, "");
static_assert(!semiregular<has_rvalue_reference_member>, "");
static_assert(!semiregular<has_function_ref_member>, "");
static_assert(!semiregular<deleted_assignment_from_const_rvalue>, "");

// Not default_initialzable
static_assert(!semiregular<no_copy_constructor>, "");
static_assert(!semiregular<no_copy_assignment>, "");
#if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017 // MSVC chokes on multiple definitions of SMF
static_assert(cuda::std::is_copy_assignable_v<no_copy_assignment_mutable>, "");
static_assert(!semiregular<no_copy_assignment_mutable>, "");
#endif // !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017
static_assert(!semiregular<derived_from_noncopyable>, "");
static_assert(!semiregular<has_noncopyable>, "");

static_assert(!semiregular<no_default_ctor>, "");
static_assert(!semiregular<derived_from_non_default_initializable>, "");
static_assert(!semiregular<has_non_default_initializable>, "");

static_assert(!semiregular<deleted_default_ctor>, "");
static_assert(!semiregular<derived_from_deleted_default_ctor>, "");
static_assert(!semiregular<has_deleted_default_ctor>, "");

int main(int, char**)
{
  return 0;
}
