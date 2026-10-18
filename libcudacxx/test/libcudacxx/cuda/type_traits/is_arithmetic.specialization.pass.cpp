//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__type_traits/is_arithmetic.h>

#include "test_macros.h"

struct UserSpecialization
{};

struct FloatingPointSpecialization
{};

template <>
constexpr bool cuda::__is_arithmetic_v<UserSpecialization> = true;

template <>
constexpr bool cuda::is_floating_point_v<FloatingPointSpecialization> = true;

TEST_FUNC void test_specialization()
{
  static_assert(cuda::__is_arithmetic_v<UserSpecialization>);
  static_assert(cuda::__is_arithmetic_v<FloatingPointSpecialization>);
  static_assert(!cuda::__is_arithmetic_v<UserSpecialization&>);
  static_assert(!cuda::__is_arithmetic_v<UserSpecialization*>);
  static_assert(!cuda::__is_arithmetic_v<FloatingPointSpecialization&>);
  static_assert(!cuda::__is_arithmetic_v<FloatingPointSpecialization*>);
}

int main(int, char**)
{
  test_specialization();
  return 0;
}
