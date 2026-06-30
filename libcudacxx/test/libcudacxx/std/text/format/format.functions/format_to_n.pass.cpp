//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// template<class Out, class... Args>
//   format_to_n_result<Out> format_to_n(Out out, iter_difference_t<Out> n,
//                                       format-string<Args...> fmt, const Args&... args);
// template<class Out, class... Args>
//   format_to_n_result<Out> format_to_n(Out out, iter_difference_t<Out> n,
//                                       wformat-string<Args...> fmt, const Args&... args);

#include <cuda/std/__format_>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

static_assert(cuda::std::is_same_v<
              cuda::std::format_to_n_result<char*>,
              decltype(cuda::std::format_to_n(cuda::std::declval<char*>(), cuda::std::iter_difference_t<char*>{}, ""))>);
static_assert(!noexcept(cuda::std::format_to_n(cuda::std::declval<char*>(), cuda::std::iter_difference_t<char*>{}, "")));
#if _CCCL_HAS_WCHAR_T()
static_assert(
  cuda::std::is_same_v<
    cuda::std::format_to_n_result<wchar_t*>,
    decltype(cuda::std::format_to_n(cuda::std::declval<wchar_t*>(), cuda::std::iter_difference_t<wchar_t*>{}, L""))>);
static_assert(
  !noexcept(cuda::std::format_to_n(cuda::std::declval<wchar_t*>(), cuda::std::iter_difference_t<wchar_t*>{}, L"")));
#endif // _CCCL_HAS_WCHAR_T()

#include "checkers/format_to_n.h"
#include "tests/smoke.h"
