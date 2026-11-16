//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/__cccl/attributes.h> must not use optional vendor attributes whose names users define as macros.

// The MSVC STL evaluates some of these names in its own feature checks.
#if __has_include(<version>)
#  include <version>
#endif // __has_include(<version>)

#define clang
#define intrinsic
#define lifetimebound
#define msvc
#define no_specializations

#include <cuda/std/__cccl/attributes.h>

static_assert(!_CCCL_HAS_ATTRIBUTE_NO_SPECIALIZATIONS());

template <class T>
struct _CCCL_NO_SPECIALIZATIONS no_specializations_test
{};

_CCCL_INTRINSIC constexpr int&& intrinsic_test(int& x) noexcept
{
  return static_cast<int&&>(x);
}

constexpr const int& lifetimebound_test(const int& x _CCCL_LIFETIMEBOUND) noexcept
{
  return x;
}

int main(int, char**)
{
  return 0;
}
