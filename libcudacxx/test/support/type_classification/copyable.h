//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef TEST_SUPPORT_TYPE_CLASSIFICATION_COPYABLE_H
#define TEST_SUPPORT_TYPE_CLASSIFICATION_COPYABLE_H

#include "movable.h"
#include "test_macros.h"

struct no_copy_constructor
{
  no_copy_constructor() = default;

  no_copy_constructor(no_copy_constructor const&) = delete;
  no_copy_constructor(no_copy_constructor&&)      = default;
};

struct no_copy_assignment
{
  no_copy_assignment() = default;

  no_copy_assignment& operator=(no_copy_assignment const&) = delete;
  no_copy_assignment& operator=(no_copy_assignment&&)      = default;
};

#if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017 // MSVC chokes on multiple definitions of SMF
struct no_copy_assignment_mutable
{
  no_copy_assignment_mutable() = default;

  no_copy_assignment_mutable& operator=(no_copy_assignment_mutable const&) = default;
  no_copy_assignment_mutable& operator=(no_copy_assignment_mutable&)       = delete;
  no_copy_assignment_mutable& operator=(no_copy_assignment_mutable&&)      = default;
};
#endif // !defined(TEST_COMPILER_MSVC) || TEST_STD_VER > 2017

struct non_copyable
{
  non_copyable() = default;
  __host__ __device__ non_copyable(non_copyable&&) {}
  __host__ __device__ non_copyable& operator=(non_copyable&&)
  {
    return *this;
  }
  non_copyable(const non_copyable&)            = delete;
  non_copyable& operator=(const non_copyable&) = delete;
};

struct derived_from_noncopyable : non_copyable
{};

struct has_noncopyable
{
  non_copyable x;
};

struct const_copy_assignment
{
  const_copy_assignment() = default;

  __host__ __device__ const_copy_assignment(const_copy_assignment const&);
  __host__ __device__ const_copy_assignment(const_copy_assignment&&);

  __host__ __device__ const_copy_assignment& operator=(const_copy_assignment&&);
  __host__ __device__ const_copy_assignment const& operator=(const_copy_assignment const&) const;
};

struct volatile_copy_assignment
{
  volatile_copy_assignment() = default;

  __host__ __device__ volatile_copy_assignment(volatile_copy_assignment volatile&);
  __host__ __device__ volatile_copy_assignment(volatile_copy_assignment volatile&&);

  __host__ __device__ volatile_copy_assignment& operator=(volatile_copy_assignment&&);
  __host__ __device__ volatile_copy_assignment volatile& operator=(volatile_copy_assignment const&) volatile;
};

struct cv_copy_assignment
{
  cv_copy_assignment() = default;

  __host__ __device__ cv_copy_assignment(cv_copy_assignment const volatile&);
  __host__ __device__ cv_copy_assignment(cv_copy_assignment const volatile&&);

  __host__ __device__ cv_copy_assignment const volatile& operator=(cv_copy_assignment const volatile&) const volatile;
  __host__ __device__ cv_copy_assignment const volatile& operator=(cv_copy_assignment const volatile&&) const volatile;
};

#endif // TEST_SUPPORT_TYPE_CLASSIFICATION_COPYABLE_H
