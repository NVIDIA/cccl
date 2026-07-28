//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/std/__format_>
#include <cuda/std/algorithm>
#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>
#include <cuda/std/string_view>
#include <cuda/std/utility>

#include "format_functions_common.h"
#include "test_macros.h"

// Marking checkers with _CCCL_NOINLINE greatly improves ptxas compile times.

template <class CharT, class... Args>
TEST_FUNC _CCCL_NOINLINE bool
check(cuda::std::basic_string_view<CharT> expected, test_format_string<CharT, Args...> fmt, Args&&... args)
{
  assert(expected.size() < 4096 && "Update the size of the buffer.");
  {
    cuda::std::inplace_vector<CharT, 4096> out;
    const auto result =
      cuda::std::format_to_n(cuda::std::__back_insert_iterator{out}, 0, fmt, cuda::std::forward<Args>(args)...);
    if (cuda::std::cmp_not_equal(result.size, expected.size()))
    {
      return false;
    }
    if (!out.empty())
    {
      return false;
    }
  }
  {
    constexpr auto n = 5;
    cuda::std::inplace_vector<CharT, 4096> out;
    const auto result =
      cuda::std::format_to_n(cuda::std::__back_insert_iterator{out}, n, fmt, cuda::std::forward<Args>(args)...);
    if (cuda::std::cmp_not_equal(result.size, expected.size()))
    {
      return false;
    }
    const auto size = cuda::std::min(cuda::std::size_t{n}, expected.size());
    if (!cuda::std::equal(out.begin(), out.end(), expected.begin(), expected.begin() + size))
    {
      return false;
    }
  }
  {
    constexpr auto n = 10;
    cuda::std::inplace_vector<CharT, 4096> out(n, CharT{' '});
    const auto result = cuda::std::format_to_n(out.begin(), n, fmt, cuda::std::forward<Args>(args)...);
    if (cuda::std::cmp_not_equal(result.size, expected.size()))
    {
      return false;
    }
    const auto size = cuda::std::min(cuda::std::size_t{n}, expected.size());
    if (result.out != out.begin() + size)
    {
      return false;
    }
    if (!cuda::std::equal(out.begin(), out.begin() + size, expected.begin(), expected.begin() + size))
    {
      return false;
    }
    if (!cuda::std::all_of(out.begin() + size, out.end(), cuda::equal_to_value{CharT{' '}}))
    {
      return false;
    }
  }
  {
    static_assert(cuda::std::is_signed_v<cuda::std::iter_difference_t<CharT*>>);
    CharT out[]{CharT{0}};
    const auto result = cuda::std::format_to_n(out, -1, fmt, cuda::std::forward<Args>(args)...);
    if (cuda::std::cmp_not_equal(result.size, expected.size()))
    {
      return false;
    }
    if (result.out != out)
    {
      return false;
    }
    if (out[0] != CharT{0})
    {
      return false;
    }
  }
  return true;
}

template <class CharT, class... Args>
TEST_FUNC bool check_exception(cuda::std::string_view, cuda::std::basic_string_view<CharT>, Args&&...)
{
  // After P2216 most exceptions thrown by std::format become ill-formed.
  // Therefore this tests does nothing.
  // A basic ill-formed test is done in format.verify.cpp
  // The exceptions are tested by other functions that don't use the basic-format-string as fmt argument.
  return true;
}
