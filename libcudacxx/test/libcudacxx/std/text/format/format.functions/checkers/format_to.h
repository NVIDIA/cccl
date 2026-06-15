//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

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
    cuda::std::inplace_vector<CharT, 4096> out(expected.size(), CharT{' '});
    auto it = cuda::std::format_to(out.begin(), fmt, cuda::std::forward<Args>(args)...);
    if (it != out.end())
    {
      return false;
    }
    if (!cuda::std::equal(out.begin(), it, expected.begin(), expected.end()))
    {
      return false;
    }
  }
  {
    cuda::std::inplace_vector<CharT, 4096> out;
    cuda::std::format_to(cuda::std::__back_insert_iterator{out}, fmt, cuda::std::forward<Args>(args)...);
    if (!cuda::std::equal(out.begin(), out.end(), expected.begin(), expected.end()))
    {
      return false;
    }
  }
  {
    CharT out[4096];
    CharT* it = cuda::std::format_to(out, fmt, cuda::std::forward<Args>(args)...);
    if (cuda::std::distance(out, it) != int(expected.size()))
    {
      return false;
    }
    if (cuda::std::basic_string_view<CharT>{out, it} != expected)
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
