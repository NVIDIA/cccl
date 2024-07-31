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
// concept regular = see below;

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "test_macros.h"
#include "type_classification/moveconstructible.h"
#include "type_classification/semiregular.h"

using cuda::std::regular;

static_assert(regular<int>, "");
static_assert(regular<float>, "");
static_assert(regular<double>, "");
static_assert(regular<long double>, "");
static_assert(regular<int volatile>, "");
static_assert(regular<void*>, "");
static_assert(regular<int*>, "");
static_assert(regular<int const*>, "");
static_assert(regular<int volatile*>, "");
static_assert(regular<int volatile const*>, "");
static_assert(regular<int (*)()>, "");

struct S
{};
static_assert(!regular<S>, "");
static_assert(regular<int S::*>, "");
static_assert(regular<int (S::*)()>, "");
static_assert(regular<int (S::*)() noexcept>, "");
static_assert(regular<int (S::*)() &>, "");
static_assert(regular<int (S::*)() & noexcept>, "");
static_assert(regular<int (S::*)() &&>, "");
static_assert(regular < int(S::*)() && noexcept >, "");
static_assert(regular<int (S::*)() const>, "");
static_assert(regular<int (S::*)() const noexcept>, "");
static_assert(regular<int (S::*)() const&>, "");
static_assert(regular<int (S::*)() const & noexcept>, "");
static_assert(regular<int (S::*)() const&&>, "");
static_assert(regular < int(S::*)() const&& noexcept >, "");
static_assert(regular<int (S::*)() volatile>, "");
static_assert(regular<int (S::*)() volatile noexcept>, "");
static_assert(regular<int (S::*)() volatile&>, "");
static_assert(regular<int (S::*)() volatile & noexcept>, "");
static_assert(regular<int (S::*)() volatile&&>, "");
static_assert(regular < int(S::*)() volatile && noexcept >, "");
static_assert(regular<int (S::*)() const volatile>, "");
static_assert(regular<int (S::*)() const volatile noexcept>, "");
static_assert(regular<int (S::*)() const volatile&>, "");
static_assert(regular<int (S::*)() const volatile & noexcept>, "");
static_assert(regular<int (S::*)() const volatile&&>, "");
static_assert(regular < int(S::*)() const volatile&& noexcept >, "");

union U
{};
static_assert(!regular<U>, "");
static_assert(regular<int U::*>, "");
static_assert(regular<int (U::*)()>, "");
static_assert(regular<int (U::*)() noexcept>, "");
static_assert(regular<int (U::*)() &>, "");
static_assert(regular<int (U::*)() & noexcept>, "");
static_assert(regular<int (U::*)() &&>, "");
static_assert(regular < int(U::*)() && noexcept >, "");
static_assert(regular<int (U::*)() const>, "");
static_assert(regular<int (U::*)() const noexcept>, "");
static_assert(regular<int (U::*)() const&>, "");
static_assert(regular<int (U::*)() const & noexcept>, "");
static_assert(regular<int (U::*)() const&&>, "");
static_assert(regular < int(U::*)() const&& noexcept >, "");
static_assert(regular<int (U::*)() volatile>, "");
static_assert(regular<int (U::*)() volatile noexcept>, "");
static_assert(regular<int (U::*)() volatile&>, "");
static_assert(regular<int (U::*)() volatile & noexcept>, "");
static_assert(regular<int (U::*)() volatile&&>, "");
static_assert(regular < int(U::*)() volatile && noexcept >, "");
static_assert(regular<int (U::*)() const volatile>, "");
static_assert(regular<int (U::*)() const volatile noexcept>, "");
static_assert(regular<int (U::*)() const volatile&>, "");
static_assert(regular<int (U::*)() const volatile & noexcept>, "");
static_assert(regular<int (U::*)() const volatile&&>, "");
static_assert(regular < int(U::*)() const volatile&& noexcept >, "");

static_assert(!regular<has_volatile_member>, "");
static_assert(!regular<has_array_member>, "");

// Not objects
static_assert(!regular<void>, "");
static_assert(!regular<int&>, "");
static_assert(!regular<int const&>, "");
static_assert(!regular<int volatile&>, "");
static_assert(!regular<int const volatile&>, "");
static_assert(!regular<int&&>, "");
static_assert(!regular<int const&&>, "");
static_assert(!regular<int volatile&&>, "");
static_assert(!regular<int const volatile&&>, "");
static_assert(!regular<int()>, "");
static_assert(!regular<int (&)()>, "");
static_assert(!regular<int[5]>, "");

// not copyable
static_assert(!regular<int const>, "");
static_assert(!regular<int const volatile>, "");
static_assert(!regular<volatile_copy_assignment volatile>, "");
static_assert(!regular<no_copy_constructor>, "");
static_assert(!regular<no_copy_assignment>, "");
#if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017 // MSVC chokes on multiple definitions of SMF
static_assert(!regular<no_copy_assignment_mutable>, "");
#endif // !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017
static_assert(!regular<derived_from_noncopyable>, "");
static_assert(!regular<has_noncopyable>, "");
static_assert(!regular<has_const_member>, "");
static_assert(!regular<has_cv_member>, "");
static_assert(!regular<has_lvalue_reference_member>, "");
static_assert(!regular<has_rvalue_reference_member>, "");
static_assert(!regular<has_function_ref_member>, "");
static_assert(!regular<deleted_assignment_from_const_rvalue>, "");

// not default_initializable
static_assert(!regular<no_copy_constructor>, "");
static_assert(!regular<no_copy_assignment>, "");
#if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017 // MSVC chokes on multiple definitions of SMF
static_assert(cuda::std::is_copy_assignable_v<no_copy_assignment_mutable> && !regular<no_copy_assignment_mutable>, "");
#endif // !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017
static_assert(!regular<derived_from_noncopyable>, "");
static_assert(!regular<has_noncopyable>, "");

static_assert(!regular<derived_from_non_default_initializable>, "");
static_assert(!regular<has_non_default_initializable>, "");

// not equality_comparable
static_assert(!regular<const_copy_assignment const>, "");
static_assert(!regular<cv_copy_assignment const volatile>, "");

struct is_equality_comparable
{
  __host__ __device__ bool operator==(is_equality_comparable const&) const
  {
    return true;
  }
  __host__ __device__ bool operator!=(is_equality_comparable const&) const
  {
    return false;
  }
};
static_assert(regular<is_equality_comparable>, "");

int main(int, char**)
{
  return 0;
}
