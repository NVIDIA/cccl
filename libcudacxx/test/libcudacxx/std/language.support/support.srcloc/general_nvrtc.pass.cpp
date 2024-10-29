//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: !nvrtc

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/source_location>
#include <cuda/std/type_traits>

#include "test_macros.h"

static_assert(cuda::std::is_nothrow_move_constructible<cuda::std::source_location>::value, "support.srcloc.cons (1.1)");
static_assert(cuda::std::is_nothrow_move_assignable<cuda::std::source_location>::value, "support.srcloc.cons (1.2)");
#if TEST_STD_VER >= 2014
static_assert(cuda::std::is_nothrow_swappable<cuda::std::source_location>::value, "support.srcloc.cons (1.3)");
#endif // TEST_STD_VER >= 2014

ASSERT_NOEXCEPT(cuda::std::source_location());
#if !defined(TEST_COMPILER_NVCC)
ASSERT_NOEXCEPT(cuda::std::source_location::current());
#endif // TEST_COMPILER_NVCC

__host__ __device__ bool compare_strings(const char* lhs, const char* rhs) noexcept
{
  for (size_t index = 0;; ++index)
  {
    if (lhs[index] != rhs[index])
    {
      return false;
    }

    if (lhs[index] == '\0')
    {
      return true;
    }
  }
}

__host__ __device__ bool find_substring(const char* source, const char* target) noexcept
{
  if (target[0] == '\0')
  {
    return true;
  }

  for (size_t index = 0;; ++index)
  {
    if (source[index] == target[index])
    {
      for (size_t sub_index = 0;; ++sub_index)
      {
        if (source[sub_index] != target[sub_index])
        {
          break;
        }

        if (target[sub_index] == '\0')
        {
          return true;
        }
      }
    }

    if (target[index] == '\0')
    {
      return false;
    }
  }
}

__device__ __constant__ cuda::std::source_location global_source = cuda::std::source_location::current();

__host__ __device__ void test()
{
  assert(!compare_strings(global_source.file_name(), ""));
  // assert(compare_strings(global_source.function_name(), "__builtin_FUNCTION is unsupported"));
  assert(global_source.line() != 0);

// nvrtc only supports this in C++20
#if TEST_STD_VER >= 2020
  assert(global_source.column() != 0);
#else
  assert(global_source.column() == 0);
#endif // TEST_STD_VER <= 2017

#line 2000
  auto local = cuda::std::source_location::current();
  assert(compare_strings(global_source.file_name(), local.file_name()));

// nvrtc only supports this in C++20
#if TEST_STD_VER >= 2020
  assert(local.line() == 2000);
#else
  assert(global_source.line() == local.line());
#endif // TEST_STD_VER <= 2017

  // This is expected
  // assert(global_source.column() == local.column());

  // Finally, the type should be copy-constructible
  auto copyied = local;
  assert(compare_strings(copyied.file_name(), local.file_name()));
  assert(compare_strings(copyied.function_name(), local.function_name()));
  assert(copyied.line() == local.line());
  assert(copyied.column() == local.column());

  // and copy-assignable.
  local = global_source;
  assert(compare_strings(local.file_name(), global_source.file_name()));
  assert(compare_strings(local.function_name(), global_source.function_name()));
  assert(local.line() == global_source.line());
  assert(local.column() == global_source.column());
}

// and inside a function.
int main(int, char**)
{
  test();
  return 0;
}
