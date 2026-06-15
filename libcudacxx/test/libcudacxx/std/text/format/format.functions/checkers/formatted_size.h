//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__format_>
#include <cuda/std/inplace_vector>
#include <cuda/std/string_view>
#include <cuda/std/utility>

#include "format_functions_common.h"
#include "test_macros.h"

// Marking checkers with _CCCL_NOINLINE greatly improves ptxas compile times.

template <class CharT, class... Args>
TEST_FUNC _CCCL_NOINLINE bool
check(cuda::std::basic_string_view<CharT> expected, test_format_string<CharT, Args...> fmt, Args&&... args)
{
  const auto size = cuda::std::formatted_size(fmt, cuda::std::forward<Args>(args)...);
  return size == expected.size();
}

template <class CharT, class... Args>
TEST_FUNC _CCCL_NOINLINE bool check_exception(cuda::std::string_view, cuda::std::basic_string_view<CharT>, Args&&...)
{
  // After P2216 most exceptions thrown by std::formatted_size become ill-formed.
  // Therefore this tests does nothing.
  // A basic ill-formed test is done in formatted_size.verify.cpp
  // The exceptions are tested by other functions that don't use the basic-format-string as fmt argument.

  return true;
}
