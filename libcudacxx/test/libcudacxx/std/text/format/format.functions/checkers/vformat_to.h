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

#include "format_functions_common.h"
#include "test_macros.h"

// Marking checkers with _CCCL_NOINLINE greatly improves ptxas compile times.

template <class CharT, class... Args>
TEST_FUNC _CCCL_NOINLINE bool
check(cuda::std::basic_string_view<CharT> expected, cuda::std::basic_string_view<CharT> fmt, Args&&... args)
{
  assert(expected.size() < 4096 && "Update the size of the buffer.");
  {
    cuda::std::inplace_vector<CharT, 4096> out;
    cuda::std::vformat_to(
      cuda::std::__back_insert_iterator{out}, fmt, cuda::std::make_format_args<context_t<CharT>>(args...));
    if (!cuda::std::equal(out.begin(), out.end(), expected.begin(), expected.end()))
    {
      return false;
    }
  }
  {
    CharT out[4096];
    CharT* it = cuda::std::vformat_to(out, fmt, cuda::std::make_format_args<context_t<CharT>>(args...));
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
TEST_FUNC _CCCL_NOINLINE bool check_exception(
  [[maybe_unused]] cuda::std::string_view what,
  [[maybe_unused]] cuda::std::basic_string_view<CharT> fmt,
  [[maybe_unused]] Args&&... args)
{
#if _CCCL_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, ({
                 try
                 {
                   cuda::std::inplace_vector<char, 4096> out;
                   cuda::std::vformat_to(cuda::std::__back_insert_iterator{out},
                                         fmt,
                                         cuda::std::make_format_args<context_t<CharT>>(args...));
                   return false;
                 }
                 catch (const cuda::std::format_error& e)
                 {
                   if (e.what() != what)
                   {
                     return false;
                   }
                 }
                 catch (...)
                 {
                   return false;
                 }
               }))
#endif // _CCCL_HAS_EXCEPTIONS()
  return true;
}
