//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <cuda/std/optional>

// Make sure we properly generate special member functions for optional<T>
// based on the properties of T itself.

#include <cuda/std/optional>
#include <cuda/std/type_traits>

#if !defined(_CCCL_BUILTIN_ADDRESSOF)
#  define TEST_WORKAROUND_NO_ADDRESSOF
#endif

#include "archetypes.h"
#include "test_macros.h"

template <class T>
struct SpecialMemberTest
{
  using O = cuda::std::optional<T>;

#ifndef TEST_COMPILER_ICC
  static_assert(cuda::std::is_default_constructible_v<O>, "optional is always default constructible.");

  static_assert(cuda::std::is_copy_constructible_v<O> == cuda::std::is_copy_constructible_v<T>,
                "optional<T> is copy constructible if and only if T is copy constructible.");

  static_assert(cuda::std::is_move_constructible_v<O>
                  == (cuda::std::is_copy_constructible_v<T> || cuda::std::is_move_constructible_v<T>),
                "optional<T> is move constructible if and only if T is copy or move constructible.");
#endif // TEST_COMPILER_ICC

  static_assert(cuda::std::is_copy_assignable_v<O>
                  == (cuda::std::is_copy_constructible_v<T> && cuda::std::is_copy_assignable_v<T>),
                "optional<T> is copy assignable if and only if T is both copy "
                "constructible and copy assignable.");

  static_assert(cuda::std::is_move_assignable_v<O>
                  == ((cuda::std::is_move_constructible_v<T> && cuda::std::is_move_assignable_v<T>)
                      || (cuda::std::is_copy_constructible_v<T> && cuda::std::is_copy_assignable_v<T>) ),
                "optional<T> is move assignable if and only if T is both move constructible and "
                "move assignable, or both copy constructible and copy assignable.");
};

template <class... Args>
static __host__ __device__ void sink(Args&&...)
{}

template <class... TestTypes>
struct DoTestsMetafunction
{
  __host__ __device__ DoTestsMetafunction()
  {
    sink(SpecialMemberTest<TestTypes>{}...);
  }
};

int main(int, char**)
{
  sink(ImplicitTypes::ApplyTypes<DoTestsMetafunction>{},
       ExplicitTypes::ApplyTypes<DoTestsMetafunction>{},
       NonLiteralTypes::ApplyTypes<DoTestsMetafunction>{},
       NonTrivialTypes::ApplyTypes<DoTestsMetafunction>{});
  return 0;
}
