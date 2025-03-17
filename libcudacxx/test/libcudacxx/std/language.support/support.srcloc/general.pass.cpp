//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// The way we currently compile nvrtc is not working with that test
// UNSUPPORTED: nvrtc && !c++20

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/source_location>
#include <cuda/std/type_traits>

#include "test_macros.h"

static_assert(cuda::std::is_nothrow_move_constructible<cuda::std::source_location>::value, "support.srcloc.cons (1.1)");
static_assert(cuda::std::is_nothrow_move_assignable<cuda::std::source_location>::value, "support.srcloc.cons (1.2)");
static_assert(cuda::std::is_nothrow_swappable<cuda::std::source_location>::value, "support.srcloc.cons (1.3)");

static_assert(noexcept(cuda::std::source_location()));
#if !TEST_CUDA_COMPILER(NVCC)
static_assert(noexcept(cuda::std::source_location::current()));
#endif // TEST_CUDA_COMPILER(NVCC)

// Note: the standard doesn't strictly require the particular values asserted
// here, but does "suggest" them.  Additional tests for details of how the
// implementation of current() chooses which location to report for more complex
// scenarios are in the Clang test-suite, and not replicated here.

// A default-constructed value.
constexpr cuda::std::source_location empty{};
static_assert(empty.line() == 0, "");
static_assert(empty.column() == 0, "");
static_assert(empty.file_name()[0] == '\0', "");
static_assert(empty.function_name()[0] == '\0', "");

static_assert(noexcept(empty.line()));
static_assert(noexcept(empty.column()));
static_assert(noexcept(empty.file_name()));
static_assert(noexcept(empty.function_name()));
static_assert(cuda::std::is_same<cuda::std::uint_least32_t, decltype(empty.line())>::value, "");
static_assert(cuda::std::is_same<cuda::std::uint_least32_t, decltype(empty.column())>::value, "");
static_assert(cuda::std::is_same<const char*, decltype(empty.file_name())>::value, "");
static_assert(cuda::std::is_same<const char*, decltype(empty.function_name())>::value, "");

__device__ constexpr cuda::std::source_location device_empty{};
static_assert(device_empty.line() == 0, "");
static_assert(device_empty.column() == 0, "");
static_assert(device_empty.file_name()[0] == '\0', "");
static_assert(device_empty.function_name()[0] == '\0', "");

static_assert(noexcept(device_empty.line()));
static_assert(noexcept(device_empty.column()));
static_assert(noexcept(device_empty.file_name()));
static_assert(noexcept(device_empty.function_name()));

// A simple use of current() outside a function.
#line 1000 "ss"
constexpr cuda::std::source_location cur = cuda::std::source_location::current();
static_assert(cur.line() == 1000, "");

#if _CCCL_HAS_BUILTIN(__builtin_COLUMN) || TEST_COMPILER(MSVC, >=, 19, 27)
static_assert(cur.column() > 0, "");
#else // ^^^ _CCCL_BULTIN_COLUMN ^^^ / vvv !_CCCL_BULTIN_COLUMN vvv
static_assert(cur.column() == 0, "");
#endif // !_CCCL_BULTIN_COLUMN
static_assert(cur.file_name()[0] == __FILE__[0] && cur.file_name()[1] == __FILE__[1]
                && cur.file_name()[sizeof(__FILE__) - 1] == '\0',
              "");

// MSVC below 19.27 is broken with function name
#if !TEST_COMPILER(MSVC, <, 19, 27)
static_assert(cur.function_name()[0] == '\0', "");
#else // ^^^ __builtin_FUNCTION ^^^ / vvv !__builtin_FUNCTION vvv
static_assert(compare_strings(cur.function_name(), "__builtin_FUNCTION is unsupported"));
#endif // !__builtin_FUNCTION

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

__host__ __device__ void test()
{
#line 2000
  auto local = cuda::std::source_location::current();
  assert(compare_strings(local.file_name(), __FILE__));

  // MSVC below 19.27 is broken with function name
#if !TEST_COMPILER(MSVC, <, 19, 27)
  assert(find_substring(local.function_name(), "test"));
#else // ^^^ __builtin_FUNCTION ^^^ / vvv !__builtin_FUNCTION vvv
  assert(compare_strings(local.function_name(), "__builtin_FUNCTION is unsupported"));
#endif // !__builtin_FUNCTION

  assert(local.line() == 2000);
#if _CCCL_HAS_BUILTIN(__builtin_COLUMN) || TEST_COMPILER(MSVC, >=, 19, 27)
  assert(cur.column() > 0);
#else // ^^^ _CCCL_BULTIN_COLUMN ^^^ / vvv !_CCCL_BULTIN_COLUMN vvv
  assert(cur.column() == 0);
#endif // !_CCCL_BULTIN_COLUMN

  // Finally, the type should be copy-constructible
  auto local2 = cur;
  assert(compare_strings(local2.file_name(), cur.file_name()));
  assert(compare_strings(local2.function_name(), cur.function_name()));
  assert(local2.line() == cur.line());
  assert(local2.column() == cur.column());

  // and copy-assignable.
  local = cur;
  assert(compare_strings(local.file_name(), cur.file_name()));
  assert(compare_strings(local.function_name(), cur.function_name()));
  assert(local.line() == cur.line());
  assert(local.column() == cur.column());
}

// and inside a function.
int main(int, char**)
{
  test();
  return 0;
}
